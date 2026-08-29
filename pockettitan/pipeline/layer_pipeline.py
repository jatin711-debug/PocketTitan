"""Layer-by-layer external memory execution pipeline with live stream progress and resumable state tracking."""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import safetensors.torch
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from pockettitan.config import (
    MemoryBudgetConfig,
    ModelMetadata,
    QuantConfig,
    QuantMethod,
    TensorAddress,
)
from pockettitan.manifest import JobManifest, ManifestManager, TensorJobRecord, TensorStatus
from pockettitan.metadata.repo import fetch_model_config
from pockettitan.metadata.tensor_index import build_tensor_address_table
from pockettitan.quantizers import get_quantizer
from pockettitan.quantizers.base import BaseQuantizer, QuantizedResult
from pockettitan.scheduler.budget import compute_work_unit_bounds
from pockettitan.scheduler.tiler import MatrixTiler
from pockettitan.streaming.reader import LocalTensorReader, RemoteTensorSliceReader


class ShardWriter:
    """Writes quantized tensors into bounded Safetensors shards continuously."""

    def __init__(self, output_dir: Union[str, Path], max_shard_size_mb: float = 2048.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = int(max_shard_size_mb * 1024 * 1024)
        
        self.current_shard_idx = 1
        self.current_buffer: Dict[str, torch.Tensor] = {}
        self.current_buffer_bytes = 0
        self.written_shards: List[str] = []
        self.weight_map: Dict[str, str] = {}
        self.total_written_bytes = 0

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Stage an unquantized or passthrough tensor (e.g. norm, bias, embedding)."""
        tensor_bytes = tensor.nbytes
        if self.current_buffer_bytes + tensor_bytes > self.max_shard_bytes and self.current_buffer:
            self.flush()
            
        self.current_buffer[name] = tensor.contiguous().cpu()
        self.current_buffer_bytes += tensor_bytes

    def add_quantized_result(self, name: str, quant_res: QuantizedResult) -> None:
        """Stage a quantized result (packed weights, scales, zeros)."""
        packed_tensors = quant_res.to_packed_tensors(name_prefix=name)
        res_bytes = quant_res.size_bytes()
        
        if self.current_buffer_bytes + res_bytes > self.max_shard_bytes and self.current_buffer:
            self.flush()
            
        for k, v in packed_tensors.items():
            self.current_buffer[k] = v.contiguous().cpu()
            
        self.current_buffer_bytes += res_bytes

    def flush(self) -> None:
        """Write current buffer to a new Safetensors shard file."""
        if not self.current_buffer:
            return
            
        shard_filename = f"model-{self.current_shard_idx:05d}.safetensors"
        shard_path = self.output_dir / shard_filename
        
        safetensors.torch.save_file(self.current_buffer, str(shard_path))
        for t_name in self.current_buffer.keys():
            self.weight_map[t_name] = shard_filename
            
        self.written_shards.append(shard_filename)
        self.total_written_bytes += self.current_buffer_bytes
        self.current_shard_idx += 1
        self.current_buffer.clear()
        self.current_buffer_bytes = 0

    def finalize(self, base_config: Optional[Dict] = None, quant_config: Optional[QuantConfig] = None) -> Dict[str, Any]:
        """Flush remaining buffer and write index.json and quant_config.json."""
        self.flush()
        
        if not self.weight_map and (self.output_dir / "model.safetensors.index.json").exists():
            with open(self.output_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
                return json.load(f)

        total_shards = max(1, len(self.written_shards))
        new_weight_map: Dict[str, str] = {}
        
        for idx, old_filename in enumerate(self.written_shards, start=1):
            new_filename = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
            old_path = self.output_dir / old_filename
            new_path = self.output_dir / new_filename
            if old_path.exists():
                old_path.rename(new_path)
            for t_name, s_name in self.weight_map.items():
                if s_name == old_filename:
                    new_weight_map[t_name] = new_filename

        self.weight_map = new_weight_map

        index_data = {
            "metadata": {
                "total_size": self.total_written_bytes,
                "quantized_by": "PocketTitan",
            },
            "weight_map": self.weight_map,
        }
        with open(self.output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        if base_config:
            with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(base_config, f, indent=2)
                
        if quant_config:
            with open(self.output_dir / "quant_config.json", "w", encoding="utf-8") as f:
                json.dump(quant_config.model_dump(), f, indent=2)

        return index_data


class QuantizationPipeline:
    """Layer-by-layer external memory execution pipeline with live stream progress."""

    def __init__(
        self,
        model_id_or_path: str,
        output_dir: Union[str, Path],
        quant_config: QuantConfig,
        budget_config: MemoryBudgetConfig,
        token: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.model_id_or_path = model_id_or_path
        self.output_dir = Path(output_dir)
        self.quant_config = quant_config
        self.budget = budget_config
        self.token = token
        self.progress_callback = progress_callback
        
        self.is_local = Path(model_id_or_path).exists() and Path(model_id_or_path).is_dir()
        if self.is_local:
            self.reader = LocalTensorReader(model_id_or_path)
        else:
            self.reader = RemoteTensorSliceReader(model_id_or_path, token=token)
            
        self.quantizer = get_quantizer(quant_config)
        self.tiler = MatrixTiler(budget_config)
        self.shard_writer = ShardWriter(output_dir, max_shard_size_mb=budget_config.max_cpu_staging_mb)
        self.console = Console()

    def _should_quantize_tensor(self, name: str, shape: List[int]) -> bool:
        """Determine if a tensor should be quantized or preserved in FP16/BF16."""
        if len(shape) <= 1:
            return False
            
        preserve_keywords = ["embed_tokens", "wte", "lm_head", "router", "gate.weight"]
        for kw in preserve_keywords:
            if kw in name and "mlp.experts" not in name:
                return False
                
        return True

    def run(self) -> Dict[str, Any]:
        """Execute complete external memory quantization pipeline with live stream progress."""
        start_time = time.time()
        self.console.print(f"[bold cyan]Starting PocketTitan Quantization Pipeline for {self.model_id_or_path}[/bold cyan]")
        
        # Build 100% address table
        table = build_tensor_address_table(self.model_id_or_path, token=self.token, fast_inspect=False)
        all_tensors = list(table.tensors.values())
        total_tensors = len(all_tensors)
        
        manifest_mgr = ManifestManager(self.output_dir)
        if manifest_mgr.exists():
            manifest = manifest_mgr.load()
            self.console.print(f"[yellow]Resuming existing job from manifest ({manifest.completed_tensors}/{manifest.total_tensors} completed)[/yellow]")
        else:
            manifest = manifest_mgr.create_initial(
                model_id_or_path=self.model_id_or_path,
                tensor_names_with_shards=[(t.name, t.shard) for t in all_tensors],
                quant_method=self.quant_config.method.value,
                bits=self.quant_config.bits,
                group_size=self.quant_config.group_size,
            )
            
        self.console.print(f"Total Tensors in Index: {total_tensors} | Execution Device: {self.quant_config.device}")
        base_config = fetch_model_config(self.model_id_or_path, token=self.token)
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        
        with progress:
            main_task = progress.add_task(f"[bold green]Overall Progress ({total_tensors} tensors)", total=total_tensors)
            stream_task = progress.add_task("[cyan]Streaming tensor...", total=100, visible=False)
            
            for idx, addr in enumerate(all_tensors, start=1):
                rec = manifest.records.get(addr.name)
                if rec and rec.status == TensorStatus.COMPLETED:
                    progress.update(main_task, description=f"[green]Skipping completed {addr.name[:25]}...", advance=1)
                    continue
                    
                progress.update(main_task, description=f"[blue]Processing ({idx}/{total_tensors}): {addr.name[:32]}...")
                
                # Setup download callback for this tensor
                expected_bytes = addr.size_bytes
                progress.reset(stream_task, total=expected_bytes, visible=True, description=f"[cyan]Streaming {addr.name[:25]}...")
                
                def on_chunk(downloaded_chunk_len: int, total_b: int):
                    progress.advance(stream_task, downloaded_chunk_len)
                    
                try:
                    if self._should_quantize_tensor(addr.name, addr.shape):
                        # Enforce tile-before-materialize invariant directly via tiler
                        q_res, peak_vram = self.tiler.quantize_address(
                            reader=self.reader,
                            tensor_addr=addr,
                            quantizer=self.quantizer,
                            target_device=self.quant_config.device,
                            chunk_callback=on_chunk if not self.is_local else None,
                        )
                        progress.update(stream_task, visible=False)
                        self.shard_writer.add_quantized_result(addr.name, q_res)
                        del q_res
                    else:
                        tensor_data = self.reader.read_tensor(addr, chunk_callback=on_chunk if not self.is_local else None)
                        progress.update(stream_task, visible=False)
                        self.shard_writer.add_tensor(addr.name, tensor_data)
                        del tensor_data
                        
                    if rec:
                        rec.status = TensorStatus.COMPLETED
                        manifest.completed_tensors += 1
                        manifest_mgr.save(manifest)
                except Exception as e:
                    if rec:
                        rec.status = TensorStatus.FAILED
                        rec.error_message = str(e)
                        manifest_mgr.save(manifest)
                    raise
                    
                progress.advance(main_task, 1)
                if self.progress_callback:
                    self.progress_callback(addr.name, idx, total_tensors)

        # Finalize and write index
        index_data = self.shard_writer.finalize(base_config=base_config, quant_config=self.quant_config)
        elapsed = time.time() - start_time
        
        if hasattr(self.reader, "close"):
            self.reader.close()
            
        self.console.print(f"[bold green]Successfully quantized {total_tensors} tensors in {elapsed:.2f}s![/bold green]")
        self.console.print(f"Output saved to: [cyan]{self.output_dir.resolve()}[/cyan]")
        return index_data
