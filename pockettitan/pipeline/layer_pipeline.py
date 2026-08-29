"""Layer-by-layer external memory streaming pipeline and incremental ShardWriter."""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import safetensors.torch
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from pockettitan.config import MemoryBudgetConfig, ModelMetadata, QuantConfig, QuantMethod, TensorAddress
from pockettitan.manifest import ManifestManager, TensorStatus
from pockettitan.metadata.repo import fetch_model_config
from pockettitan.metadata.tensor_index import TensorAddressTable, build_tensor_address_table
from pockettitan.quantizers import get_quantizer
from pockettitan.quantizers.base import QuantizedResult
from pockettitan.scheduler.tiler import MatrixTiler
from pockettitan.streaming.reader import LocalTensorReader, RemoteTensorSliceReader


class ShardWriter:
    """Incrementally writes quantized tensors to Safetensors files with strict memory cap."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        max_shard_size_mb: float = 2048.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_size_mb * 1024 * 1024
        
        self.current_shard_idx = 1
        self.current_buffer: Dict[str, torch.Tensor] = {}
        self.current_buffer_bytes = 0
        self.weight_map: Dict[str, str] = {}
        self.written_shards: List[str] = []
        self.total_written_bytes = 0

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Add tensor to staging buffer, flushing shard to disk if size limit reached."""
        tensor_cpu = tensor.cpu().contiguous()
        t_bytes = tensor_cpu.nbytes
        
        if self.current_buffer_bytes + t_bytes > self.max_shard_bytes and self.current_buffer:
            self.flush()
            
        self.current_buffer[name] = tensor_cpu
        self.current_buffer_bytes += t_bytes

    def add_quantized_result(self, name_prefix: str, result: QuantizedResult) -> None:
        """Store packed weights, scales, and zeros with standard naming convention."""
        base_name = name_prefix[:-7] if name_prefix.endswith(".weight") else name_prefix
        self.add_tensor(f"{base_name}.packed_weight", result.packed_weights)
        self.add_tensor(f"{base_name}.scales", result.scales)
        if result.zeros is not None:
            self.add_tensor(f"{base_name}.zeros", result.zeros)
        if result.codebook is not None:
            self.add_tensor(f"{base_name}.codebook", result.codebook)

    def flush(self) -> None:
        """Write current in-memory buffer to disk as a Safetensors file."""
        if not self.current_buffer:
            return
            
        shard_filename = f"model-{self.current_shard_idx:05d}.safetensors"
        shard_path = self.output_dir / shard_filename
        
        safetensors.torch.save_file(self.current_buffer, str(shard_path))
        
        for name in self.current_buffer.keys():
            self.weight_map[name] = shard_filename
            
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

        # Rename shards with total count: model-00001-of-00005.safetensors
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

        # Write model.safetensors.index.json
        index_data = {
            "metadata": {
                "total_size": self.total_written_bytes,
                "quantized_by": "PocketTitan",
            },
            "weight_map": self.weight_map,
        }
        with open(self.output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        # Write config.json
        if base_config:
            with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(base_config, f, indent=2)
                
        # Write quant_config.json
        if quant_config:
            with open(self.output_dir / "quant_config.json", "w", encoding="utf-8") as f:
                json.dump(quant_config.model_dump(), f, indent=2)

        return index_data


class QuantizationPipeline:
    """Layer-by-layer external memory execution pipeline."""

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
        # 1. 1D tensors (biases, norms, scales) are preserved in full precision
        if len(shape) <= 1:
            return False
            
        # 2. Embeddings, LM heads, and MoE routers preserved by default for stability
        preserve_keywords = ["embed_tokens", "wte", "lm_head", "router", "gate.weight"]
        # If user explicitly requested all linear layers
        for kw in preserve_keywords:
            if kw in name and "mlp.experts" not in name:
                return False
                
        return True

    def run(self) -> Dict[str, Any]:
        """Execute complete external memory quantization pipeline with resume support."""
        start_time = time.time()
        self.console.print(f"[bold cyan]Starting PocketTitan Quantization Pipeline for {self.model_id_or_path}[/bold cyan]")
        
        table = build_tensor_address_table(self.model_id_or_path, token=self.token)
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
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("Quantizing model...", total=total_tensors)
            
            for idx, addr in enumerate(all_tensors, start=1):
                rec = manifest.records.get(addr.name)
                if rec and rec.status == TensorStatus.COMPLETED:
                    progress.update(task_id, description=f"Skipping completed {addr.name[:25]}...", advance=1)
                    continue
                    
                progress.update(task_id, description=f"Processing {addr.name[:35]}...", advance=1)
                
                # Check if tensor should be quantized
                if self._should_quantize_tensor(addr.name, addr.shape):
                    tensor_data = self.reader.read_tensor(addr)
                    
                    q_res, peak_vram = self.tiler.quantize_matrix(
                        tensor_data,
                        quantizer=self.quantizer,
                        target_device=self.quant_config.device,
                    )
                    
                    self.shard_writer.add_quantized_result(addr.name, q_res)
                    del tensor_data, q_res
                else:
                    tensor_data = self.reader.read_tensor(addr)
                    self.shard_writer.add_tensor(addr.name, tensor_data)
                    del tensor_data
                    
                if rec:
                    rec.status = TensorStatus.COMPLETED
                    manifest.completed_tensors += 1
                    manifest_mgr.save(manifest)
                    
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
