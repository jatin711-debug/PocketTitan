"""PocketTitan Command Line Interface."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import torch
import typer
from rich import print as rprint
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from pockettitan import __version__
from pockettitan.audit import Capability, build_audit_report, get_precision_preset, scan_checkpoint
from pockettitan.audit.report import render_report
from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod, parse_memory_to_mb
from pockettitan.export.validator import CheckpointValidator
from pockettitan.exporters.gguf import GGUFExporter
from pockettitan.exporters.vllm import VLLMExporter
from pockettitan.metadata.repo import fetch_model_config, inspect_model_repository
from pockettitan.metadata.tensor_index import build_tensor_address_table
from pockettitan.models.moe import parse_moe_layer_structure
from pockettitan.package import PackageWriter, plan_package
from pockettitan.package.report import render_plan
from pockettitan.pipeline.layer_pipeline import QuantizationPipeline
from pockettitan.precision.allocator import ParetoBitAllocator
from pockettitan.precision.distortion import evaluate_quantization_quality
from pockettitan.quantizers import get_quantizer
from pockettitan.scheduler.budget import apply_cuda_memory_fraction, compute_work_unit_bounds, get_hardware_profile
from pockettitan.scheduler.tiler import MatrixTiler
from pockettitan.streaming.reader import LocalTensorReader, RemoteTensorSliceReader

app = typer.Typer(
    name="pockettitan",
    help="External-memory post-training quantization engine for extreme-scale LLMs.",
    add_completion=False,
)


def _configure_stdio_encoding() -> None:
    """Make stdout/stderr safe for the report glyphs.

    Windows consoles default to a legacy codepage (cp1252) that cannot encode
    the box-drawing and status characters Rich emits, which crashes mid-render
    once output is piped. Prefer UTF-8; degrade to replacement characters rather
    than raising.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


_configure_stdio_encoding()

# When output is redirected Rich falls back to 80 columns, which truncates
# 12-digit parameter counts to "120,795,9...". An audit that silently elides
# digits is not an audit, so widen non-interactive output.
console = Console(width=None if sys.stdout.isatty() else 160)


def format_size(num_bytes: float) -> str:
    """Format bytes into human-readable GiB/MiB/KiB."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"


def format_params(num_params: int) -> str:
    """Format parameter count into Millions / Billions."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f} B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f} M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f} K"
    return str(num_params)


@app.command()
def version():
    """Print PocketTitan version."""
    console.print(f"[bold cyan]PocketTitan[/bold cyan] version [green]{__version__}[/green] [yellow](Alpha / Research & Development Preview - Not for Production)[/yellow]")


@app.command()
def hardware():
    """Scan and display local hardware capabilities and memory limits."""
    hw = get_hardware_profile()
    
    table = Table(title="[bold green]Local Hardware Profile[/bold green]", show_header=True)
    table.add_column("Resource", style="cyan", no_wrap=True)
    table.add_column("Details", style="white")
    table.add_column("Available / Capacity", style="magenta")
    
    if hw.cuda_available:
        for dev in hw.devices:
            table.add_row(
                f"GPU #{dev.device_id}",
                f"{dev.name} (Compute {dev.compute_capability[0]}.{dev.compute_capability[1]})",
                f"{dev.free_vram_mb:.0f} MiB free / {dev.total_vram_mb:.0f} MiB total",
            )
    else:
        table.add_row("GPU", "None detected", "CPU Mode Only")
        
    table.add_row(
        "Host RAM",
        "System Memory",
        f"{hw.system_ram_free_mb:.0f} MiB free / {hw.system_ram_total_mb:.0f} MiB total",
    )
    table.add_row(
        "Storage",
        "Target Working Disk",
        f"{hw.disk_free_mb / 1024:.2f} GiB free",
    )
    console.print(table)


@app.command()
def test_matrix(
    out_features: int = typer.Option(4096, "--out-features", "-m", help="Matrix row dimension"),
    in_features: int = typer.Option(4096, "--in-features", "-k", help="Matrix column dimension"),
    method: QuantMethod = typer.Option(QuantMethod.HQQ, "--method", help="Quantization algorithm (hqq, rtn, ternary, intx, gptq, awq, autoround)"),
    bits: int = typer.Option(2, "--bits", "-b", help="Bit-width (1, 2, 3, 4, 8)"),
    group_size: int = typer.Option(128, "--group-size", "-g", help="Group size for groupwise quantization"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Hard VRAM budget ceiling (e.g. '2GB', '1500MB', '4GiB', '2048')"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to execute on (cuda/cpu)"),
):
    """Milestones 1 & 2: Benchmark single matrix and micro-tiler under strict VRAM caps."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    console.print(f"[bold cyan]Benchmarking Matrix Quantization:[/bold cyan] [{out_features} x {in_features}] | Method: {method.value} ({bits}-bit) | User VRAM Cap: {max_vram_mb:.0f} MiB ({max_vram})")
    
    budget = MemoryBudgetConfig(max_vram_mb=max_vram_mb)
    apply_cuda_memory_fraction(budget)
    quant_cfg = QuantConfig(method=method, bits=bits, group_size=group_size, device=device)
    quantizer = get_quantizer(quant_cfg)
    tiler = MatrixTiler(budget)
    
    torch.manual_seed(42)
    w_orig = torch.randn(out_features, in_features, dtype=torch.float16, device="cpu") * 0.02
    
    quant_res, peak_vram_mb = tiler.quantize_matrix(
        w_orig, quantizer=quantizer, target_device=device
    )
    
    w_deq = quantizer.dequantize(quant_res).cpu()
    report = evaluate_quantization_quality(w_orig, w_deq)
    
    raw_bytes = w_orig.nbytes
    quant_bytes = quant_res.size_bytes()
    compression_ratio = raw_bytes / max(1, quant_bytes)
    
    table = Table(title="[bold green]Matrix Quantization & Memory Report[/bold green]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Matrix Dimensions", f"{out_features} x {in_features} ({out_features * in_features:,} params)")
    table.add_row("Original Size (FP16)", format_size(raw_bytes))
    table.add_row("Quantized Size", f"{format_size(quant_bytes)} ({compression_ratio:.2f}x compression)")
    table.add_row("Effective Bit-width", f"{quant_res.bit_width:.2f} bits/weight")
    table.add_row("User-Specified VRAM Cap", f"{max_vram_mb:.0f} MiB ({max_vram})")
    table.add_row("Measured Peak CUDA VRAM", f"[bold magenta]{peak_vram_mb:.2f} MiB[/bold magenta]")
    table.add_row("Relative Weight Distortion (Frobenius)", f"{report.weight_distortion:.6f}")
    table.add_row("Signal-to-Noise Ratio (SNR)", f"{report.snr_db:.2f} dB")
    table.add_row("Cosine Similarity", f"{report.cosine_similarity:.6f}")
    
    vram_status = "[bold green]PASS (Strictly Under Budget)[/bold green]" if peak_vram_mb <= max_vram_mb else "[bold red]FAIL (Exceeded Budget)[/bold red]"
    table.add_row("VRAM Enforcement Status", vram_status)
    console.print(table)


@app.command()
def optimize_precision(
    model: str = typer.Argument(..., help="Model ID (e.g. Qwen/Qwen1.5-MoE-A2.7B) or local directory"),
    target_bpw: float = typer.Option(2.2, "--target-bpw", "-b", help="Target average bits-per-weight across model"),
    output_map: str = typer.Option("precision_map.json", "--output", "-o", help="Output path for precision assignment JSON"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
):
    """Milestone 8: Solve Pareto-optimal heterogeneous precision allocation map."""
    console.print(f"[bold cyan]Solving Pareto Precision Map for {model} (Target: {target_bpw:.2f} bpw)...[/bold cyan]")
    table_idx = build_tensor_address_table(model, token=token)
    all_tensors = list(table_idx.tensors.values())
    
    allocator = ParetoBitAllocator(target_bpw=target_bpw)
    pmap = allocator.solve(model, all_tensors, table_idx.metadata)
    
    with open(output_map, "w", encoding="utf-8") as f:
        f.write(pmap.model_dump_json(indent=2))
        
    table = Table(title="[bold green]Pareto Precision Optimization Summary[/bold green]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Target Average Bits/Weight", f"{target_bpw:.2f} bpw")
    table.add_row("Effective Average Bits/Weight", f"[bold magenta]{pmap.effective_bpw:.2f} bpw[/bold magenta]")
    table.add_row("Total Tensors Mapped", str(len(pmap.tensor_quant_configs)))
    table.add_row("Estimated Compression Ratio", f"{16.0 / max(0.1, pmap.effective_bpw):.2f}x vs FP16")
    table.add_row("Saved Map To", output_map)
    console.print(table)


@app.command()
def export(
    checkpoint_dir: str = typer.Argument(..., help="Directory containing quantized Safetensors checkpoint"),
    output: str = typer.Option(..., "--output", "-o", help="Target output file (for GGUF) or directory (for vLLM)"),
    format: str = typer.Option("gguf", "--format", "-f", help="Target format: gguf or vllm"),
):
    """Milestone 9: Export quantized model checkpoint to GGUF (llama.cpp) or vLLM format."""
    console.print(f"[bold cyan]Exporting checkpoint {checkpoint_dir} to format: {format.upper()}...[/bold cyan]")
    
    validator = CheckpointValidator(checkpoint_dir)
    scorecard = validator.validate()
    if not scorecard.is_valid:
        console.print(f"[bold red]Checkpoint validation failed. Cannot export.[/bold red]")
        for err in scorecard.errors:
            console.print(f" - {err}")
        raise typer.Exit(code=1)
        
    if format.lower() == "gguf":
        exporter = GGUFExporter(checkpoint_dir)
    elif format.lower() == "vllm":
        exporter = VLLMExporter(checkpoint_dir)
    else:
        console.print(f"[bold red]Unsupported format: {format}. Use 'gguf' or 'vllm'.[/bold red]")
        raise typer.Exit(code=1)
        
    res = exporter.export(output)
    
    table = Table(title="[bold green]Export Completion Report[/bold green]", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Export Format", res.format_name.upper())
    table.add_row("Destination", res.output_path)
    table.add_row("Total Tensors Exported", str(res.total_tensors))
    table.add_row("Final Output Size", format_size(res.output_size_bytes))
    table.add_row("Status", "[bold green]SUCCESS[/bold green]")
    console.print(table)


@app.command()
def quantize_expert(
    intermediate_size: int = typer.Option(2048, "--intermediate-size", "-i", help="MoE expert intermediate dimension (e.g. 2048 for DeepSeek)"),
    hidden_size: int = typer.Option(7168, "--hidden-size", "-k", help="Model hidden dimension (e.g. 7168 for DeepSeek)"),
    method: QuantMethod = typer.Option(QuantMethod.HQQ, "--method", "-m", help="Quantization algorithm"),
    bits: int = typer.Option(2, "--bits", "-b", help="Bit-width"),
    group_size: int = typer.Option(128, "--group-size", "-g", help="Group size"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Hard VRAM budget ceiling (e.g. '2GB', '1500MB', '4GiB')"),
    device: str = typer.Option("cuda", "--device", "-d", help="Execution device"),
):
    """Milestone 4: Quantize all 3 projection matrices of a single MoE expert under strict VRAM caps."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    console.print(f"[bold cyan]Benchmarking Single MoE Expert Quantization:[/bold cyan] Intermediate: {intermediate_size}, Hidden: {hidden_size} | Method: {method.value} ({bits}-bit) | VRAM Cap: {max_vram_mb:.0f} MiB ({max_vram})")
    
    budget = MemoryBudgetConfig(max_vram_mb=max_vram_mb)
    apply_cuda_memory_fraction(budget)
    quant_cfg = QuantConfig(method=method, bits=bits, group_size=group_size, device=device)
    quantizer = get_quantizer(quant_cfg)
    tiler = MatrixTiler(budget)
    
    torch.manual_seed(42)
    w_gate = torch.randn(intermediate_size, hidden_size, dtype=torch.float16, device="cpu") * 0.02
    w_up = torch.randn(intermediate_size, hidden_size, dtype=torch.float16, device="cpu") * 0.02
    w_down = torch.randn(hidden_size, intermediate_size, dtype=torch.float16, device="cpu") * 0.02
    
    total_raw_bytes = w_gate.nbytes + w_up.nbytes + w_down.nbytes
    peak_vram_mb = 0.0
    
    q_gate, p1 = tiler.quantize_matrix(w_gate, quantizer=quantizer, target_device=device)
    q_up, p2 = tiler.quantize_matrix(w_up, quantizer=quantizer, target_device=device)
    q_down, p3 = tiler.quantize_matrix(w_down, quantizer=quantizer, target_device=device)
    peak_vram_mb = max(p1, p2, p3)
    
    total_quant_bytes = q_gate.size_bytes() + q_up.size_bytes() + q_down.size_bytes()
    compression_ratio = total_raw_bytes / max(1, total_quant_bytes)
    
    w_down_deq = quantizer.dequantize(q_down).cpu()
    report = evaluate_quantization_quality(w_down, w_down_deq)
    
    table = Table(title="[bold green]Single MoE Expert Quantization Report[/bold green]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    total_params = (intermediate_size * hidden_size * 2) + (hidden_size * intermediate_size)
    table.add_row("Total Expert Parameters", f"{total_params:,} ({format_params(total_params)})")
    table.add_row("Original Size (FP16)", format_size(total_raw_bytes))
    table.add_row("Quantized Size", f"{format_size(total_quant_bytes)} ({compression_ratio:.2f}x compression)")
    table.add_row("User-Specified VRAM Cap", f"{max_vram_mb:.0f} MiB ({max_vram})")
    table.add_row("Measured Peak CUDA VRAM", f"[bold magenta]{peak_vram_mb:.2f} MiB[/bold magenta]")
    table.add_row("Down Proj Cosine Sim", f"{report.cosine_similarity:.6f}")
    table.add_row("Down Proj SNR", f"{report.snr_db:.2f} dB")
    
    vram_status = "[bold green]PASS (Strictly Under Budget)[/bold green]" if peak_vram_mb <= max_vram_mb else "[bold red]FAIL (Exceeded Budget)[/bold red]"
    table.add_row("VRAM Enforcement Status", vram_status)
    console.print(table)


@app.command()
def quantize(
    model: str = typer.Argument(..., help="Model ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0) or local path"),
    output_dir: str = typer.Option("./quantized_model", "--output-dir", "-o", help="Target output folder"),
    method: QuantMethod = typer.Option(QuantMethod.HQQ, "--method", "-m", help="Quantization algorithm (hqq, ternary, rtn, intx, gptq, awq, autoround)"),
    bits: int = typer.Option(2, "--bits", "-b", help="Bit-width (1, 2, 3, 4, 8)"),
    group_size: int = typer.Option(128, "--group-size", "-g", help="Group size for groupwise quantization"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Hard peak VRAM ceiling (e.g. '2GB', '4GB', '1500MB', '2048')"),
    max_cpu_staging: str = typer.Option("2048MB", "--max-cpu-staging", help="Max staging buffer before shard flush (e.g. '2GB', '1024MB')"),
    device: str = typer.Option("cuda", "--device", "-d", help="Execution device (cuda/cpu)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
):
    """Milestone 3 & 6: Stream and quantize a model under strict user VRAM cap."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    max_staging_mb = parse_memory_to_mb(max_cpu_staging)
    
    budget = MemoryBudgetConfig(
        max_vram_mb=max_vram_mb,
        max_cpu_staging_mb=max_staging_mb,
    )
    apply_cuda_memory_fraction(budget)
    
    quant_cfg = QuantConfig(
        method=method,
        bits=bits,
        group_size=group_size,
        device=device,
    )
    
    pipeline = QuantizationPipeline(
        model_id_or_path=model,
        output_dir=output_dir,
        quant_config=quant_cfg,
        budget_config=budget,
        token=token,
    )
    pipeline.run()


@app.command()
def validate(
    checkpoint_dir: str = typer.Argument(..., help="Directory containing quantized Safetensors model"),
):
    """Milestone 7: Validate checkpoint structural and mathematical integrity."""
    console.print(f"[bold cyan]Auditing checkpoint integrity in: {checkpoint_dir}[/bold cyan]")
    validator = CheckpointValidator(checkpoint_dir)
    scorecard = validator.validate()
    
    table = Table(title="[bold green]Checkpoint Integrity Scorecard[/bold green]", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Shards Found", str(scorecard.total_shards))
    table.add_row("Total Tensors Indexed", str(scorecard.total_tensors))
    table.add_row("Quantized Tensors", str(scorecard.quantized_tensor_count))
    table.add_row("Passthrough Tensors (FP16/BF16)", str(scorecard.passthrough_tensor_count))
    table.add_row("Errors Detected", str(len(scorecard.errors)))
    table.add_row("Warnings", str(len(scorecard.warnings)))
    
    status_str = "[bold green]PASS (Valid Checkpoint)[/bold green]" if scorecard.is_valid else "[bold red]FAIL (Corrupted/Incomplete)[/bold red]"
    table.add_row("Overall Validity Status", status_str)
    console.print(table)
    
    if scorecard.errors:
        for err in scorecard.errors:
            console.print(f"[bold red]Error:[/bold red] {err}")
    if scorecard.warnings:
        for warn in scorecard.warnings:
            console.print(f"[bold yellow]Warning:[/bold yellow] {warn}")


@app.command()
def inspect(
    model: str = typer.Argument(..., help="Hugging Face repo ID (e.g. zai-org/GLM-5.3-Flash) or local directory path"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Hard VRAM budget ceiling (e.g. '2GB', '3.5GB', '4GiB')"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of formatted tables"),
):
    """Milestone 0: Inspect model repository, architecture, parameter count, shards, and work unit bounds."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    if not json_output:
        console.print(f"[bold green]Inspecting model metadata for [cyan]{model}[/cyan] (VRAM Cap: {max_vram_mb:.0f} MiB / {max_vram})...[/bold green]")
        
    try:
        table_idx = build_tensor_address_table(model, token=token)
        meta = table_idx.metadata
        hw = get_hardware_profile()
        budget = MemoryBudgetConfig(max_vram_mb=max_vram_mb)
    except Exception as e:
        console.print(f"[bold red]Failed to inspect model {model}:[/bold red] {e}")
        raise typer.Exit(code=1)

    if json_output:
        console.print_json(meta.model_dump_json())
        return

    # Architecture Overview Panel
    arch_info = f"""[bold cyan]Architecture:[/bold cyan] {meta.architecture}
[bold cyan]Source Precision:[/bold cyan] {meta.source_dtype}
[bold cyan]Layers:[/bold cyan] {meta.num_hidden_layers} | [bold cyan]Hidden Size:[/bold cyan] {meta.hidden_size} | [bold cyan]Attention Heads:[/bold cyan] {meta.num_attention_heads}
[bold cyan]Total Parameters:[/bold cyan] [bold green]{format_params(meta.total_params)}[/bold green] ({meta.total_params:,})
[bold cyan]Active Parameters / Token:[/bold cyan] [bold green]{format_params(meta.active_params)}[/bold green] ({meta.active_params:,})
[bold cyan]Total Shards:[/bold cyan] {len(meta.shards)} Safetensors shards | [bold cyan]Total Tensors:[/bold cyan] {len(meta.tensors)}"""

    if meta.is_moe:
        arch_info += f"""\n[bold yellow]MoE Routing:[/bold yellow] {meta.num_experts} routed experts (Top-{meta.num_experts_per_tok} active/tok) | Expert Dim: {meta.expert_intermediate_size}"""
        if meta.shared_expert_intermediate_size:
            arch_info += f" | Shared Expert Dim: {meta.shared_expert_intermediate_size}"

    console.print(Panel(arch_info, title=f"[bold green]PocketTitan Inspector: {model}[/bold green]", border_style="cyan"))

    # Theoretical Storage Footprint Table
    fp16_bytes = meta.total_params * 2
    footprint_table = Table(title="[bold]Estimated Model Storage Footprint by Precision[/bold]", show_header=True)
    footprint_table.add_column("Precision / Format", style="cyan")
    footprint_table.add_column("Effective bpw", style="yellow")
    footprint_table.add_column("Theoretical Model Size", style="green")
    footprint_table.add_column("PocketTitan Streaming Peak VRAM", style="magenta")

    footprint_table.add_row("FP16 / BF16 (Uncompressed)", "16.0", format_size(fp16_bytes), "N/A (Streamed)")
    footprint_table.add_row("FP8", "8.0", format_size(meta.total_params * 1), f"< {max_vram_mb:.0f} MiB")
    footprint_table.add_row("INT4 / HQQ4", "4.0", format_size(meta.total_params * 0.5), f"< {max_vram_mb:.0f} MiB")
    footprint_table.add_row("INT3 / HQQ3", "3.0", format_size(meta.total_params * 0.375), f"< {max_vram_mb:.0f} MiB")
    footprint_table.add_row("INT2 / HQQ2", "2.0", format_size(meta.total_params * 0.25), f"< {max_vram_mb:.0f} MiB")
    footprint_table.add_row("BitNet / Ternary W1.58", "1.58", format_size(meta.total_params * 0.20), f"< {max_vram_mb:.0f} MiB")
    console.print(footprint_table)

    # Top 5 Largest Tensors & Tiling Sizing
    largest = table_idx.largest_tensors(top_n=5)
    tensor_table = Table(title=f"[bold]Largest Tensors & Hardware Work Unit Decomposition (< {max_vram_mb:.0f} MiB Cap)[/bold]", show_header=True)
    tensor_table.add_column("Tensor Name", style="cyan", max_width=45, overflow="ellipsis")
    tensor_table.add_column("Shape", style="white")
    tensor_table.add_column("Parameters", style="green")
    tensor_table.add_column("Raw Size", style="yellow")
    tensor_table.add_column("Tiling Strategy", style="magenta")

    quant_cfg = QuantConfig(method=QuantMethod.HQQ, bits=2)
    for t in largest:
        bounds = compute_work_unit_bounds(t.shape, budget, quant_cfg, meta.source_dtype)
        if bounds.get("needs_tiling", False):
            tiling_desc = f"[red]Tiled:[/red] {bounds['num_tiles']} tiles ({bounds['tile_rows']} rows/tile, ~{bounds['estimated_vram_per_tile_mb']:.0f}MB VRAM)"
        else:
            tiling_desc = f"[green]Single Pass[/green] (~{bounds['estimated_vram_per_tile_mb']:.0f}MB VRAM)"
        tensor_table.add_row(
            t.name,
            str(t.shape),
            format_params(t.num_params),
            format_size(t.size_bytes),
            tiling_desc,
        )
    console.print(tensor_table)
@app.command()
def inspect_layer(
    checkpoint_dir: str = typer.Argument(..., help="Path to PocketTitan quantized checkpoint directory"),
    layer: int = typer.Option(0, "--layer", "-l", help="Layer index to inspect"),
):
    """Inspect and verify dequantized layer numerical fidelity and stats."""
    from pockettitan.inference import PocketTitanModelRunner
    runner = PocketTitanModelRunner(checkpoint_dir)
    res = runner.inspect_layer_sample(layer)
    console.print(f"[bold green]Inspection Result:[/bold green] {res}")


@app.command()
def plan(
    model: str = typer.Argument(..., help="Hugging Face repo ID (e.g. Qwen/Qwen3.8-Flash-Next) or local directory path"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    precision: str = typer.Option("pt-q4e", "--precision", "-p", help="Precision preset: pt-q4e, pt-q2e, bf16, int8, int4, int3, int2, ternary"),
    features: str = typer.Option("text", "--features", help="Comma-separated capabilities to keep: text,vision,mtp"),
    alignment: int = typer.Option(4096, "--expert-alignment", help="Expert record pitch in bytes; page alignment keeps reads page-aligned"),
    workers: int = typer.Option(16, "--workers", help="Parallel shard header requests"),
    json_output: bool = typer.Option(False, "--json", help="Emit the plan as JSON instead of tables"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the plan JSON to this path"),
):
    """R1: Plan a PocketTitan package — capability filter, expert repacking, precision, and PLE row store.

    Reads only checkpoint headers. No weights are fetched, so the byte-exact
    layout can be reviewed before a multi-hundred-gigabyte build starts.
    """
    try:
        selected = [Capability(f.strip().lower()) for f in features.split(",") if f.strip()]
    except ValueError:
        console.print(
            f"[bold red]Invalid --features '{features}'.[/bold red] Valid values: "
            + ", ".join(c.value for c in Capability)
        )
        raise typer.Exit(code=1)

    try:
        precision_map = get_precision_preset(precision)
    except KeyError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    if not json_output:
        console.print(
            f"[bold green]Planning package for [cyan]{model}[/cyan][/bold green] "
            f"[dim](precision={precision_map.name}, features={','.join(c.value for c in selected)})[/dim]"
        )

    try:
        with console.status("[green]Reading shard headers...", spinner="dots") if not json_output else nullcontext():
            scan = scan_checkpoint(model, token=token, max_workers=workers, strict=True)
        build_plan = plan_package(
            scan,
            precision_map=precision_map,
            features=selected,
            expert_alignment=alignment,
            pockettitan_version=__version__,
        )
    except Exception as e:
        console.print(f"[bold red]Planning failed for {model}:[/bold red] {escape(str(e))}")
        raise typer.Exit(code=1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_plan.model_dump_json(indent=2), encoding="utf-8")

    if json_output:
        console.print_json(build_plan.model_dump_json())
    else:
        render_plan(console, build_plan)
        if output is not None:
            console.print(f"\n[dim]Plan written to {output}[/dim]")


@app.command()
def package(
    model: str = typer.Argument(..., help="Hugging Face repo ID or local directory path"),
    output: Path = typer.Argument(..., help="Destination .ptitan directory"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    precision: str = typer.Option("pt-q4e", "--precision", "-p", help="Precision preset"),
    features: str = typer.Option("text", "--features", help="Comma-separated capabilities to keep"),
    method: QuantMethod = typer.Option(QuantMethod.RTN, "--method", help="Quantizer backend"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Hard VRAM ceiling"),
    device: str = typer.Option("cuda", "--device", "-d", help="Execution device (cuda/cpu)"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Rebuild from scratch, ignoring the journal"),
    workers: int = typer.Option(16, "--workers", help="Parallel shard header requests"),
):
    """R1: Build a PocketTitan package — capability-filtered, repacked, quantized.

    Resumable: rerun the same command after an interruption and only unfinished
    work items are redone.
    """
    try:
        selected = [Capability(f.strip().lower()) for f in features.split(",") if f.strip()]
        precision_map = get_precision_preset(precision)
    except (ValueError, KeyError) as e:
        console.print(f"[bold red]{escape(str(e))}[/bold red]")
        raise typer.Exit(code=1)

    budget = MemoryBudgetConfig(max_vram_mb=parse_memory_to_mb(max_vram))
    console.print(
        f"[bold green]Packaging [cyan]{model}[/cyan] -> [cyan]{output}[/cyan][/bold green] "
        f"[dim](precision={precision_map.name}, method={method.value}, VRAM cap {budget.max_vram_mb:.0f} MiB)[/dim]"
    )

    try:
        with console.status("[green]Reading shard headers...", spinner="dots"):
            scan = scan_checkpoint(model, token=token, max_workers=workers, strict=True)
        build_plan = plan_package(
            scan, precision_map=precision_map, features=selected, pockettitan_version=__version__
        )
    except Exception as e:
        console.print(f"[bold red]Planning failed:[/bold red] {escape(str(e))}")
        raise typer.Exit(code=1)

    render_plan(console, build_plan)
    for warning in build_plan.warnings:
        console.print(f"[yellow]! {escape(warning)}[/yellow]")

    if scan.is_local:
        reader = LocalTensorReader(model)
    else:
        reader = RemoteTensorSliceReader(model, token=token)

    writer = PackageWriter(
        build_plan, output, reader, budget=budget, method=method,
        device=device, resume=not no_resume,
    )

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

    console.print()
    try:
        with Progress(
            TextColumn("[cyan]{task.description}"), BarColumn(),
            TaskProgressColumn(), TimeRemainingColumn(), console=console,
        ) as progress:
            task = progress.add_task("building", total=build_plan.num_work_items)
            result = writer.build(on_item=lambda region, label: progress.update(task, advance=1))
    except Exception as e:
        console.print(f"[bold red]Build failed:[/bold red] {escape(str(e))}")
        console.print("[dim]Rerun the same command to resume from the journal.[/dim]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]Package written to {output}[/bold green]\n"
        f"  items written {result.items_written:,} · skipped {result.items_skipped:,}\n"
        f"  bytes written {format_size(result.bytes_written)}\n"
        f"  peak VRAM {result.peak_vram_mb:.1f} MiB / {budget.usable_vram_mb:.0f} MiB usable\n"
        f"  elapsed {result.elapsed_s:.1f} s"
    )


@app.command()
def audit(
    model: str = typer.Argument(..., help="Hugging Face repo ID (e.g. Qwen/Qwen3.8-Flash-Next) or local directory path"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    precision: str = typer.Option("pt-q4e", "--precision", "-p", help="Precision preset: pt-q4e, pt-q2e, bf16, int8, int4, int3, int2, ternary"),
    features: str = typer.Option("text", "--features", help="Comma-separated capabilities to keep: text,vision,mtp"),
    ram_budget: str = typer.Option("7GB", "--ram-budget", help="RAM available for the expert cache (drives roofline slot count)"),
    workers: int = typer.Option(16, "--workers", help="Parallel shard header requests"),
    no_strict: bool = typer.Option(False, "--no-strict", help="Continue past unreadable shards instead of failing"),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON instead of tables"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the report JSON to this path"),
):
    """R0: Audit a checkpoint — component decomposition, activated params/token, storage, state, and SSD roofline."""
    try:
        selected = [Capability(f.strip().lower()) for f in features.split(",") if f.strip()]
    except ValueError:
        console.print(
            f"[bold red]Invalid --features '{features}'.[/bold red] Valid values: "
            + ", ".join(c.value for c in Capability)
        )
        raise typer.Exit(code=1)

    if not selected:
        console.print("[bold red]--features must name at least one capability.[/bold red]")
        raise typer.Exit(code=1)

    try:
        precision_map = get_precision_preset(precision)
    except KeyError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    ram_budget_bytes = parse_memory_to_mb(ram_budget) * 1024.0 * 1024.0

    if not json_output:
        console.print(
            f"[bold green]Auditing [cyan]{model}[/cyan][/bold green] "
            f"[dim](precision={precision_map.name}, features={','.join(c.value for c in selected)})[/dim]"
        )

    try:
        with console.status("[green]Reading shard headers...", spinner="dots") if not json_output else nullcontext():
            scan = scan_checkpoint(model, token=token, max_workers=workers, strict=not no_strict)
        report = build_audit_report(
            scan, precision_map=precision_map, features=selected, ram_budget_bytes=ram_budget_bytes
        )
    except Exception as e:
        console.print(f"[bold red]Audit failed for {model}:[/bold red] {escape(str(e))}")
        raise typer.Exit(code=1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if json_output:
        console.print_json(report.model_dump_json())
    else:
        render_report(console, report)
        if output is not None:
            console.print(f"\n[dim]Report written to {output}[/dim]")

    if report.discrepancies:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
