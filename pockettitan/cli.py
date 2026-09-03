"""PocketTitan Command Line Interface."""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import torch
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from pockettitan import __version__
from pockettitan.audit import Capability, build_audit_report, get_precision_preset, scan_checkpoint
from pockettitan.audit.report import render_report
from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod, parse_memory_to_mb
from pockettitan.domainslice import (
    CompositeWeightStore,
    ModelRevision,
    PocketTitanPageStore,
    RemoteHuggingFaceStore,
    WeightPageID,
    generate_olmoe_text,
)
from pockettitan.domainslice.hypothesis import (
    run_olmoe_block_hypothesis,
    run_olmoe_full_model_hypothesis,
    run_olmoe_full_model_reference,
    run_olmoe_layer_hypothesis,
    run_olmoe_two_position_generation,
)
from pockettitan.export.validator import CheckpointValidator
from pockettitan.exporters.gguf import GGUFExporter
from pockettitan.exporters.vllm import VLLMExporter
from pockettitan.metadata.tensor_index import build_tensor_address_table
from pockettitan.metadata.repo import resolve_model_revision
from pockettitan.package import PackageWriter, PtitanValidator, plan_package
from pockettitan.package.report import render_plan
from pockettitan.pipeline.layer_pipeline import QuantizationPipeline
from pockettitan.precision.allocator import ParetoBitAllocator
from pockettitan.precision.distortion import evaluate_quantization_quality
from pockettitan.quantizers import get_quantizer
from pockettitan.scheduler.budget import (
    apply_cuda_memory_fraction,
    compute_work_unit_bounds,
    get_hardware_profile,
)
from pockettitan.scheduler.tiler import MatrixTiler
from pockettitan.streaming.reader import LocalTensorReader, RemoteTensorSliceReader

app = typer.Typer(
    name="pockettitan",
    help="External-memory post-training quantization engine for extreme-scale LLMs.",
    add_completion=False,
)
domainslice_app = typer.Typer(
    name="domainslice",
    help="Inspect and demand-page routed experts from immutable checkpoints.",
    no_args_is_help=True,
)
app.add_typer(domainslice_app, name="domainslice")


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


@domainslice_app.command("inspect")
def domainslice_inspect(
    model: str = typer.Argument(..., help="Hugging Face model ID"),
    revision: Optional[str] = typer.Option(
        None, "--revision", help="Branch, tag, or immutable checkpoint commit"
    ),
):
    """Resolve an MoE checkpoint and inspect its virtual expert address space."""
    token = os.environ.get("HF_TOKEN")
    try:
        with console.status("[cyan]Resolving immutable model revision..."):
            commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        remote = RemoteHuggingFaceStore(model_revision, token=token, max_workers=3)
        with console.status("[cyan]Reading Safetensors headers (no weight payloads)..."):
            address_table = remote.address_table
    except Exception as exc:
        console.print(f"[bold red]DomainSlice inspection failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    metadata = address_table.metadata
    summary = Table(title="[bold green]DomainSlice Model Address Space[/bold green]")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Repository", model)
    summary.add_row("Immutable revision", commit_sha)
    summary.add_row("Architecture", metadata.architecture)
    summary.add_row("MoE", "yes" if metadata.is_moe else "no")
    summary.add_row("Layers", str(metadata.num_hidden_layers))
    summary.add_row("Experts/layer", str(metadata.num_experts or 0))
    summary.add_row("Experts/token", str(metadata.num_experts_per_tok or 0))
    summary.add_row("Expert width", str(metadata.expert_intermediate_size or 0))
    summary.add_row("Checkpoint shards", str(len(metadata.shards)))
    summary.add_row("Addressed tensors", f"{len(address_table.tensors):,}")
    summary.add_row("Addressed payload", format_size(address_table.total_bytes))
    summary.add_row("Authentication", "HF_TOKEN" if token else "anonymous")
    console.print(summary)
    if not metadata.is_moe:
        console.print(
            "[yellow]This is a dense model. Use sequential layer streaming; routed-expert "
            "demand paging does not apply.[/yellow]"
        )


@domainslice_app.command("fetch-expert")
def domainslice_fetch_expert(
    model: str = typer.Argument(..., help="Hugging Face MoE model ID"),
    layer: int = typer.Option(..., "--layer", min=0, help="Transformer layer index"),
    expert: int = typer.Option(..., "--expert", min=0, help="Routed expert index"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Local NVMe page-cache directory"),
    revision: Optional[str] = typer.Option(
        None, "--revision", help="Branch, tag, or immutable checkpoint commit"
    ),
    download_workers: int = typer.Option(
        3,
        "--download-workers",
        min=1,
        max=32,
        help="Parallel projection-range downloads",
    ),
    max_cache: str = typer.Option(
        "50GB", "--max-cache", help="Maximum completed local page-cache size"
    ),
):
    """Fetch one routed expert as an interruption-safe immutable local page."""
    token = os.environ.get("HF_TOKEN")
    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    store = None
    handle = None
    try:
        console.print("[cyan]1/5[/cyan] Resolving immutable model revision")
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        page_id = WeightPageID.expert(model_revision, layer, expert)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        store = CompositeWeightStore(local, remote, download_workers=download_workers)

        console.print("[cyan]2/5[/cyan] Looking up the verified local page")
        local_descriptor = local.resolve(page_id)
        if local_descriptor is None:
            console.print("[cyan]3/5[/cyan] Reading checkpoint headers and resolving exact ranges")
        else:
            console.print("[cyan]3/5[/cyan] Reusing the cached page descriptor")
        descriptor = store.resolve(page_id)

        ranges = Table(title=f"Expert page — layer {layer}, expert {expert}")
        ranges.add_column("Projection", style="cyan")
        ranges.add_column("Shard")
        ranges.add_column("Byte range", justify="right")
        ranges.add_column("Payload", justify="right")
        for item in descriptor.source_slices:
            ranges.add_row(
                item.projection,
                item.shard,
                f"{item.byte_start:,}..{item.byte_end - 1:,}",
                format_size(item.size_bytes),
            )
        console.print(ranges)

        console.print("[cyan]4/5[/cyan] Materializing and verifying the expert page")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress_bar:
            task = progress_bar.add_task("Exact expert payload", total=descriptor.expected_bytes)

            def report(_stage: str, _item: str, count: int, _total: int) -> None:
                progress_bar.advance(task, count)

            handle = store.materialize(page_id, progress=report)
            progress_bar.update(task, completed=descriptor.expected_bytes)

        stats = store.stats()
        console.print("[cyan]5/5[/cyan] Page committed and cache accounting updated")
        result = Table(title="[bold green]DomainSlice Fetch Result[/bold green]")
        result.add_column("Metric", style="cyan")
        result.add_column("Value", style="white")
        result.add_row("Repository", model)
        result.add_row("Immutable revision", commit_sha)
        result.add_row("Page", page_id.logical_key)
        result.add_row("Cache", "HIT" if handle.cache_hit else "MISS")
        result.add_row("Requested payload", format_size(descriptor.expected_bytes))
        result.add_row("Network payload", format_size(handle.bytes_fetched))
        result.add_row("Reused partial bytes", format_size(handle.bytes_resumed))
        result.add_row("Page size", format_size(handle.size_bytes))
        result.add_row("SHA-256", handle.checksum)
        result.add_row("Cache occupancy", format_size(stats.cache_occupancy_bytes))
        result.add_row("Cached pages", str(stats.cached_pages))
        result.add_row("Elapsed", f"{handle.timings.get('total_seconds', 0.0):.3f} s")
        result.add_row("Path", str(handle.path.resolve()))
        console.print(result)
    except Exception as exc:
        console.print(f"[bold red]DomainSlice fetch failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            if handle is not None:
                store.release(handle)
            store.close()


@domainslice_app.command("test-block")
def domainslice_test_block(
    model: str = typer.Argument(..., help="Hugging Face OLMoE model ID"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Local NVMe page-cache directory"),
    layer: int = typer.Option(9, "--layer", min=0, help="MoE layer to test"),
    tokens: int = typer.Option(1, "--tokens", min=1, max=16, help="Synthetic hidden-state tokens"),
    seed: int = typer.Option(42, "--seed", help="Deterministic hidden-state seed"),
    device: str = typer.Option(
        "cpu", "--device", help="Expert execution device: cpu, cuda, or auto"
    ),
    revision: Optional[str] = typer.Option(
        None, "--revision", help="Branch, tag, or immutable checkpoint commit"
    ),
    download_workers: int = typer.Option(
        3, "--download-workers", min=1, max=32, help="Parallel expert projection downloads"
    ),
    max_cache: str = typer.Option(
        "2GB", "--max-cache", help="Maximum completed local page-cache size"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional JSON path for the measured result"
    ),
):
    """Compare one real page-backed OLMoE block against Transformers."""
    token = os.environ.get("HF_TOKEN")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu, cuda, or auto")
    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    store = None
    try:
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        store = CompositeWeightStore(local, remote, download_workers=download_workers)
        console.print(
            f"[bold cyan]Testing OLMoE layer {layer}[/bold cyan] on {device}; "
            "the first run faults routed experts and the second must be local."
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress_bar:
            task = progress_bar.add_task("Remote expert payload", total=None)

            def report(_stage: str, _item: str, count: int, _total: int) -> None:
                progress_bar.advance(task, count)

            result = run_olmoe_block_hypothesis(
                store,
                remote,
                layer=layer,
                tokens=tokens,
                seed=seed,
                execution_device=device,
                progress=report,
            )

        routing = Table(title="Routing and residency")
        routing.add_column("Metric", style="cyan")
        routing.add_column("First run", justify="right")
        routing.add_column("Warm run", justify="right")
        routing.add_row("Experts", ", ".join(map(str, result.selected_experts)), "same")
        routing.add_row("Page faults", str(result.cold.page_faults), str(result.warm.page_faults))
        routing.add_row("Page hits", str(result.cold.page_hits), str(result.warm.page_hits))
        routing.add_row(
            "Remote expert bytes",
            format_size(result.cold.remote_bytes),
            format_size(result.warm.remote_bytes),
        )
        routing.add_row(
            "Runtime",
            f"{result.cold.elapsed_seconds:.3f} s",
            f"{result.warm.elapsed_seconds:.3f} s",
        )
        routing.add_row(
            "Peak staged projection",
            format_size(result.cold.peak_projection_bytes),
            format_size(result.warm.peak_projection_bytes),
        )
        console.print(routing)

        parity = Table(title="Transformers parity")
        parity.add_column("Metric", style="cyan")
        parity.add_column("Value", justify="right")
        parity.add_row("Result", "PASS" if result.passed else "FAIL")
        parity.add_row("Max absolute error", f"{result.max_abs_error:.8f}")
        parity.add_row("Mean absolute error", f"{result.mean_abs_error:.8f}")
        parity.add_row("Cosine similarity", f"{result.cosine_similarity:.8f}")
        parity.add_row("Argmax agreement", f"{result.argmax_agreement:.2%}")
        parity.add_row("Router payload", format_size(result.router_payload_bytes))
        parity.add_row("Cache occupancy", format_size(result.cache_occupancy_bytes))
        if device == "cuda":
            parity.add_row("Peak CUDA allocation", format_size(result.cold.peak_cuda_bytes))
        console.print(parity)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Measured result written to[/green] {output.resolve()}")
        if not result.passed:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]DomainSlice block test failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            store.close()


@domainslice_app.command("test-layer")
def domainslice_test_layer(
    model: str = typer.Argument(..., help="Hugging Face OLMoE model ID"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Local NVMe page-cache directory"),
    layer: int = typer.Option(9, "--layer", min=0, help="Decoder layer to test"),
    seed: int = typer.Option(42, "--seed", help="Deterministic hidden-state seed"),
    device: str = typer.Option(
        "cpu", "--device", help="Layer execution device: cpu, cuda, or auto"
    ),
    revision: Optional[str] = typer.Option(
        None, "--revision", help="Branch, tag, or immutable checkpoint commit"
    ),
    download_workers: int = typer.Option(
        3, "--download-workers", min=1, max=32, help="Parallel range downloads"
    ),
    max_cache: str = typer.Option("2GB", "--max-cache", help="Maximum local page cache"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional JSON path for the measured result"
    ),
):
    """Assemble and run one complete OLMoE decoder layer from pages."""
    token = os.environ.get("HF_TOKEN")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu, cuda, or auto")
    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    store = None
    try:
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        store = CompositeWeightStore(local, remote, download_workers=download_workers)
        console.print(
            f"[bold cyan]Testing complete OLMoE decoder layer {layer}[/bold cyan] on {device}"
        )
        result = run_olmoe_layer_hypothesis(
            store,
            remote,
            layer=layer,
            seed=seed,
            execution_device=device,
        )

        table = Table(title="DomainSlice complete-layer hypothesis")
        table.add_column("Metric", style="cyan")
        table.add_column("First", justify="right")
        table.add_column("Warm", justify="right")
        table.add_row("Result", "PASS" if result.passed else "FAIL", "deterministic")
        table.add_row(
            "Backbone pages",
            f"{result.backbone.page_faults} faults / {result.backbone.page_hits} hits",
            "resident",
        )
        table.add_row(
            "Backbone payload", format_size(result.backbone.remote_bytes), "0 B"
        )
        table.add_row(
            "Expert pages",
            f"{result.first.page_faults} faults / {result.first.page_hits} hits",
            f"{result.warm.page_faults} faults / {result.warm.page_hits} hits",
        )
        table.add_row(
            "Expert payload",
            format_size(result.first.remote_bytes),
            format_size(result.warm.remote_bytes),
        )
        table.add_row(
            "Expert runtime",
            f"{result.first.elapsed_seconds:.3f} s",
            f"{result.warm.elapsed_seconds:.3f} s",
        )
        table.add_row(
            "Peak staged projection",
            format_size(result.first.peak_projection_bytes),
            format_size(result.warm.peak_projection_bytes),
        )
        if device == "cuda":
            table.add_row(
                "Peak CUDA allocation",
                format_size(result.first.peak_cuda_bytes),
                format_size(result.warm.peak_cuda_bytes),
            )
        table.add_row(
            "Total first remote payload", format_size(result.total_first_remote_bytes), "0 B"
        )
        table.add_row(
            "Warm output delta", f"{result.warm_max_abs_delta:.8f}", "bit-identical"
        )
        table.add_row("Selected experts", ", ".join(map(str, result.selected_experts)), "same")
        table.add_row("Cache occupancy", format_size(result.cache_occupancy_bytes), "same")
        console.print(table)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Measured result written to[/green] {output.resolve()}")
        if not result.passed:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]DomainSlice layer test failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            store.close()


@domainslice_app.command("test-model")
def domainslice_test_model(
    model: str = typer.Argument(..., help="Hugging Face OLMoE model ID"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Local NVMe page-cache directory"),
    input_token_id: int = typer.Option(1, "--input-token-id", min=0),
    device: str = typer.Option(
        "auto", "--device", help="Sequential execution device: cpu, cuda, or auto"
    ),
    revision: Optional[str] = typer.Option(
        None, "--revision", help="Branch, tag, or immutable checkpoint commit"
    ),
    download_workers: int = typer.Option(
        3, "--download-workers", min=1, max=32, help="Parallel range downloads"
    ),
    max_cache: str = typer.Option("4GB", "--max-cache", help="Maximum local page cache"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Measured CUDA budget"),
    max_ram: str = typer.Option("12GB", "--max-ram", help="Measured process RSS budget"),
    head_chunk: str = typer.Option(
        "8MB", "--head-chunk", help="Maximum LM-head projection staged at once"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional JSON path for the measured result"
    ),
):
    """Run one real token through every OLMoE layer and require warm logits."""
    token = os.environ.get("HF_TOKEN")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu, cuda, or auto")
    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    max_vram_mb = parse_memory_to_mb(max_vram)
    max_ram_mb = parse_memory_to_mb(max_ram)
    head_chunk_bytes = int(parse_memory_to_mb(head_chunk) * 1024 * 1024)
    max_vram_bytes = int(max_vram_mb * 1024 * 1024)
    max_ram_bytes = int(max_ram_mb * 1024 * 1024)
    if device == "cuda":
        apply_cuda_memory_fraction(MemoryBudgetConfig(max_vram_mb=max_vram_mb))
    store = None
    try:
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        total_layers = remote.address_table.metadata.num_hidden_layers
        store = CompositeWeightStore(local, remote, download_workers=download_workers)
        console.print(
            f"[bold cyan]Testing complete OLMoE model[/bold cyan] on {device}; "
            f"input token {input_token_id}, VRAM cap {format_size(max_vram_bytes)}, "
            f"RAM cap {format_size(max_ram_bytes)}"
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress_bar:
            task = progress_bar.add_task("Discovering checkpoint", total=None)

            def report(_stage: str, _item: str, count: int, _total: int) -> None:
                progress_bar.advance(task, count)

            def layer_report(phase: str, item) -> None:
                progress_bar.update(
                    task,
                    description=(
                        f"{phase}: layer {item.layer + 1}/{total_layers} "
                        f"({item.experts.page_faults} expert faults, "
                        f"{item.elapsed_seconds:.1f}s)"
                    ),
                )

            result = run_olmoe_full_model_hypothesis(
                store,
                remote,
                input_token_id=input_token_id,
                execution_device=device,
                head_chunk_bytes=head_chunk_bytes,
                max_vram_bytes=max_vram_bytes,
                max_ram_bytes=max_ram_bytes,
                progress=report,
                layer_callback=layer_report,
            )

        table = Table(title="DomainSlice end-to-end one-token hypothesis")
        table.add_column("Metric", style="cyan")
        table.add_column("First", justify="right")
        table.add_column("Warm", justify="right")
        table.add_row("Result", "PASS" if result.passed else "FAIL", "exact replay")
        table.add_row(
            "Runtime",
            f"{result.first.elapsed_seconds:.3f} s",
            f"{result.warm.elapsed_seconds:.3f} s",
        )
        table.add_row(
            "Remote payload",
            format_size(result.first.total_remote_bytes),
            format_size(result.warm.total_remote_bytes),
        )
        table.add_row(
            "Global pages",
            f"{result.first.global_page_faults} faults / {result.first.global_page_hits} hits",
            f"{result.warm.global_page_faults} faults / {result.warm.global_page_hits} hits",
        )
        table.add_row(
            "Backbone pages",
            f"{result.first.backbone_page_faults} faults / "
            f"{result.first.backbone_page_hits} hits",
            f"{result.warm.backbone_page_faults} faults / "
            f"{result.warm.backbone_page_hits} hits",
        )
        table.add_row(
            "Expert pages",
            f"{result.first.expert_page_faults} faults / {result.first.expert_page_hits} hits",
            f"{result.warm.expert_page_faults} faults / {result.warm.expert_page_hits} hits",
        )
        table.add_row(
            "Logical page bytes",
            format_size(result.first.logical_page_bytes),
            format_size(result.warm.logical_page_bytes),
        )
        table.add_row(
            "Peak staged projection",
            format_size(result.first.peak_projection_bytes),
            format_size(result.warm.peak_projection_bytes),
        )
        table.add_row(
            "Peak staged LM head",
            format_size(result.first.peak_head_chunk_bytes),
            format_size(result.warm.peak_head_chunk_bytes),
        )
        table.add_row(
            "Peak process RSS",
            format_size(result.first.peak_rss_bytes),
            format_size(result.warm.peak_rss_bytes),
        )
        if device == "cuda":
            table.add_row(
                "Peak CUDA allocation",
                format_size(result.first.peak_cuda_bytes),
                format_size(result.warm.peak_cuda_bytes),
            )
        table.add_row(
            "Top token",
            f"{result.first.top_token_id} ({result.first.top_logit:.5f})",
            f"{result.warm.top_token_id} ({result.warm.top_logit:.5f})",
        )
        table.add_row(
            "Logit delta", f"{result.logits_max_abs_delta:.8f}", "bit-identical"
        )
        table.add_row("Cache occupancy", format_size(result.cache_occupancy_bytes), "same")
        console.print(table)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Measured result written to[/green] {output.resolve()}")
        if not result.passed:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]DomainSlice model test failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            store.close()


@domainslice_app.command("test-reference")
def domainslice_test_reference(
    model: str = typer.Argument(..., help="Hugging Face OLMoE model ID"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Local NVMe page-cache directory"),
    input_token_id: int = typer.Option(1, "--input-token-id", min=0),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    download_workers: int = typer.Option(3, "--download-workers", min=1, max=32),
    max_cache: str = typer.Option("4GB", "--max-cache"),
    max_vram: str = typer.Option("3584MB", "--max-vram"),
    max_ram: str = typer.Option("12GB", "--max-ram"),
    head_chunk: str = typer.Option("8MB", "--head-chunk"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
):
    """Compare a complete paged forward with official Transformers expert math."""
    token = os.environ.get("HF_TOKEN")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu, cuda, or auto")
    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    max_vram_mb = parse_memory_to_mb(max_vram)
    max_ram_mb = parse_memory_to_mb(max_ram)
    head_chunk_bytes = int(parse_memory_to_mb(head_chunk) * 1024 * 1024)
    max_vram_bytes = int(max_vram_mb * 1024 * 1024)
    max_ram_bytes = int(max_ram_mb * 1024 * 1024)
    if device == "cuda":
        apply_cuda_memory_fraction(MemoryBudgetConfig(max_vram_mb=max_vram_mb))
    store = None
    try:
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        total_layers = remote.address_table.metadata.num_hidden_layers
        store = CompositeWeightStore(local, remote, download_workers=download_workers)
        console.print(
            "[bold cyan]Comparing PocketTitan with official Transformers experts[/bold cyan]"
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress_bar:
            task = progress_bar.add_task("Preparing reference run", total=None)

            def report(_stage: str, _item: str, count: int, _total: int) -> None:
                progress_bar.advance(task, count)

            def layer_report(phase: str, item) -> None:
                progress_bar.update(
                    task,
                    description=(
                        f"{phase}: layer {item.layer + 1}/{total_layers} "
                        f"({item.elapsed_seconds:.1f}s)"
                    ),
                )

            result = run_olmoe_full_model_reference(
                store,
                remote,
                input_token_id=input_token_id,
                execution_device=device,
                head_chunk_bytes=head_chunk_bytes,
                max_vram_bytes=max_vram_bytes,
                max_ram_bytes=max_ram_bytes,
                progress=report,
                layer_callback=layer_report,
            )

        table = Table(title="DomainSlice independent expert-path parity")
        table.add_column("Metric", style="cyan")
        table.add_column("PocketTitan", justify="right")
        table.add_column("Transformers oracle", justify="right")
        table.add_row("Result", "PASS" if result.passed else "FAIL", "reference")
        table.add_row(
            "Runtime",
            f"{result.candidate.elapsed_seconds:.3f} s",
            f"{result.oracle.elapsed_seconds:.3f} s",
        )
        table.add_row(
            "Remote payload",
            format_size(result.candidate.total_remote_bytes),
            format_size(result.oracle.total_remote_bytes),
        )
        table.add_row(
            "Peak CUDA allocation",
            format_size(result.candidate.peak_cuda_bytes),
            format_size(result.oracle.peak_cuda_bytes),
        )
        table.add_row(
            "Peak sampled RSS",
            format_size(result.candidate.peak_rss_bytes),
            format_size(result.oracle.peak_rss_bytes),
        )
        table.add_row("Maximum error", f"{result.max_abs_error:.8f}", "0 is exact")
        table.add_row("Mean error", f"{result.mean_abs_error:.8f}", "0 is exact")
        table.add_row("Cosine similarity", f"{result.cosine_similarity:.8f}", "1 is exact")
        table.add_row("Argmax agreement", str(result.argmax_agreement), "required")
        table.add_row("Routing agreement", str(result.routing_agreement), "required")
        table.add_row("Bit exact", str(result.bit_exact), "strongest gate")
        console.print(table)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Measured result written to[/green] {output.resolve()}")
        if not result.passed:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]DomainSlice reference test failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            store.close()


@domainslice_app.command("test-generate")
def domainslice_test_generate(
    model: str = typer.Argument(..., help="Hugging Face OLMoE model ID"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Local NVMe page-cache directory"),
    input_token_id: int = typer.Option(1, "--input-token-id", min=0),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    download_workers: int = typer.Option(3, "--download-workers", min=1, max=32),
    max_cache: str = typer.Option("4GB", "--max-cache"),
    max_vram: str = typer.Option("3584MB", "--max-vram"),
    max_ram: str = typer.Option("12GB", "--max-ram"),
    head_chunk: str = typer.Option("8MB", "--head-chunk"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
):
    """Generate across two KV-cached positions and compare with the HF oracle."""
    token = os.environ.get("HF_TOKEN")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu, cuda, or auto")
    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    max_vram_mb = parse_memory_to_mb(max_vram)
    max_ram_mb = parse_memory_to_mb(max_ram)
    head_chunk_bytes = int(parse_memory_to_mb(head_chunk) * 1024 * 1024)
    max_vram_bytes = int(max_vram_mb * 1024 * 1024)
    max_ram_bytes = int(max_ram_mb * 1024 * 1024)
    if device == "cuda":
        apply_cuda_memory_fraction(MemoryBudgetConfig(max_vram_mb=max_vram_mb))
    store = None
    try:
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        total_layers = remote.address_table.metadata.num_hidden_layers
        store = CompositeWeightStore(local, remote, download_workers=download_workers)
        console.print(
            "[bold cyan]Testing two-position KV-cached OLMoE generation[/bold cyan]"
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress_bar:
            task = progress_bar.add_task("Preparing generation", total=None)

            def report(_stage: str, _item: str, count: int, _total: int) -> None:
                progress_bar.advance(task, count)

            def layer_report(phase: str, item) -> None:
                progress_bar.update(
                    task,
                    description=(
                        f"{phase}: layer {item.layer + 1}/{total_layers} "
                        f"({item.elapsed_seconds:.1f}s)"
                    ),
                )

            result = run_olmoe_two_position_generation(
                store,
                remote,
                input_token_id=input_token_id,
                execution_device=device,
                head_chunk_bytes=head_chunk_bytes,
                max_vram_bytes=max_vram_bytes,
                max_ram_bytes=max_ram_bytes,
                progress=report,
                layer_callback=layer_report,
            )

        table = Table(title="DomainSlice two-position generation")
        table.add_column("Position", justify="right")
        table.add_column("Token transition", justify="right")
        table.add_column("PocketTitan", justify="right")
        table.add_column("HF oracle", justify="right")
        table.add_column("Parity", justify="right")
        for position in result.positions:
            table.add_row(
                str(position.position),
                f"{position.input_token_id} -> {position.predicted_token_id}",
                f"{position.candidate.elapsed_seconds:.3f}s / "
                f"{format_size(position.candidate.total_remote_bytes)} remote",
                f"{position.oracle.elapsed_seconds:.3f}s",
                "exact" if position.bit_exact else f"err {position.max_abs_error:.6f}",
            )
        console.print(table)
        summary = Table(title="Generation residency")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")
        summary.add_row("Result", "PASS" if result.passed else "FAIL")
        summary.add_row("Token IDs", " -> ".join(map(str, result.token_ids)))
        summary.add_row("New remote payload", format_size(result.candidate_new_remote_bytes))
        summary.add_row("Final KV cache", format_size(result.final_kv_cache_bytes))
        summary.add_row("Cache occupancy", format_size(result.cache_occupancy_bytes))
        summary.add_row(
            "Peak PocketTitan CUDA",
            format_size(max(item.candidate.peak_cuda_bytes for item in result.positions)),
        )
        summary.add_row(
            "Peak sampled RSS",
            format_size(max(item.candidate.peak_rss_bytes for item in result.positions)),
        )
        console.print(summary)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Measured result written to[/green] {output.resolve()}")
        if not result.passed:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]DomainSlice generation test failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            store.close()


@domainslice_app.command("generate")
def domainslice_generate(
    model: str = typer.Argument(
        "allenai/OLMoE-1B-7B-0924-Instruct",
        help="Hugging Face OLMoE model ID",
    ),
    prompt: str = typer.Option(
        "Explain what a memory hierarchy is in simple terms.",
        "--prompt",
        "-p",
        help="Input text prompt to generate from",
    ),
    cache_dir: Path = typer.Option(
        Path("./olmoe-cache"),
        "--cache-dir",
        help="Local NVMe page-cache directory",
    ),
    max_new_tokens: int = typer.Option(32, "--max-new-tokens", "-n", min=1, max=1024),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="0 = greedy, >0 = sampling"),
    top_p: float = typer.Option(1.0, "--top-p", help="Nucleus sampling top-p"),
    chat: bool = typer.Option(True, "--chat/--raw", help="Apply official tokenizer chat template"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    download_workers: int = typer.Option(3, "--download-workers", min=1, max=32),
    max_cache: str = typer.Option("14GB", "--max-cache", help="Local NVMe page-cache budget ceiling"),
    max_vram: str = typer.Option("3584MB", "--max-vram"),
    head_chunk: str = typer.Option("8MB", "--head-chunk"),
    fast: bool = typer.Option(True, "--fast/--low-vram", help="Enable resident backbone & in-memory expert cache for 20x-50x speed"),
    vram_experts: int = typer.Option(64, "--vram-experts", help="Max experts resident in GPU VRAM"),
    ram_experts: int = typer.Option(384, "--ram-experts", help="Max experts resident in Host RAM"),
    quantize_ram: bool = typer.Option(False, "--quantize-ram", "--int4-cache", help="Compress Host RAM experts to 4-bit INT4 (3.3 MB each) to fit 100% of model in memory"),
    quant_bits: int = typer.Option(4, "--quant-bits", help="Bit width for RAM expert compression (e.g. 4)"),
    commit_routing: bool = typer.Option(False, "--commit-routing", help="Commit to VRAM-resident experts if gating affinity delta <= threshold (CommitMoE)"),
    commit_threshold: float = typer.Option(0.15, "--commit-threshold", help="Max gating affinity gap to commit to VRAM expert"),
    speculative: bool = typer.Option(False, "--speculative", help="Enable S2-MoE self-speculative Top-1 drafting and verification for 2x-4x speedup"),
    spec_k: int = typer.Option(3, "--spec-k", help="Speculative draft lookahead window length (default 3)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional JSON telemetry output path"),
):
    """Generate text from a natural language prompt using on-demand DomainSlice expert paging."""
    token = os.environ.get("HF_TOKEN")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu, cuda, or auto")

    max_cache_bytes = int(parse_memory_to_mb(max_cache) * 1024 * 1024)
    max_vram_mb = parse_memory_to_mb(max_vram)
    head_chunk_bytes = int(parse_memory_to_mb(head_chunk) * 1024 * 1024)

    if device == "cuda":
        apply_cuda_memory_fraction(MemoryBudgetConfig(max_vram_mb=max_vram_mb))

    store = None
    try:
        commit_sha = resolve_model_revision(model, token=token, revision=revision)
        model_revision = ModelRevision(repo_id=model, commit_sha=commit_sha)
        local = PocketTitanPageStore(cache_dir, max_cache_bytes=max_cache_bytes)
        remote = RemoteHuggingFaceStore(
            model_revision,
            token=token,
            max_workers=download_workers,
        )
        store = CompositeWeightStore(local, remote, download_workers=download_workers)

        mode_desc = "FAST-PATH" if fast else "LOW-VRAM"
        quant_desc = f" · INT{quant_bits}-RAM" if quantize_ram else ""
        commit_desc = " · COMMIT" if commit_routing else ""
        spec_desc = f" · SPEC-K{spec_k}" if speculative else ""
        console.print(
            f"[bold cyan]DomainSlice On-Demand Generation: {model}[/bold cyan] "
            f"([green]{device.upper()}[/green] · [yellow]{mode_desc}{quant_desc}{commit_desc}{spec_desc}[/yellow])"
        )
        console.print(f"[bold yellow]Prompt:[/bold yellow] {prompt}")
        sys.stdout.write("Generated Response: ")
        sys.stdout.flush()

        def stream_chunk(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        result = generate_olmoe_text(
            store,
            remote,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            chat=chat,
            execution_device=device,
            head_chunk_bytes=head_chunk_bytes,
            stream_callback=stream_chunk,
            resident_backbone=fast,
            vram_expert_capacity=vram_experts,
            ram_expert_capacity=ram_experts,
            quantize_ram=quantize_ram,
            quant_bits=quant_bits,
            commit_routing=commit_routing,
            commit_threshold=commit_threshold,
            speculative=speculative,
            spec_k=spec_k,
        )

        console.print("\n")
        summary = Table(title="DomainSlice Generation Scorecard")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")
        for line in result.summary_lines():
            name, _, val = line.partition(":")
            if val:
                summary.add_row(name.strip(), val.strip())
            else:
                summary.add_row("Detail", name.strip())
        console.print(summary)

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result.__dict__, indent=2, default=str), encoding="utf-8")
            console.print(f"[green]Telemetry written to[/green] {output.resolve()}")

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]DomainSlice generation failed:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        if store is not None:
            store.close()


@app.command()
def version():
    """Print PocketTitan version."""
    console.print(
        f"[bold cyan]PocketTitan[/bold cyan] version [green]{__version__}[/green] [yellow](Alpha / Research & Development Preview - Not for Production)[/yellow]"
    )


@app.command()
def hardware():
    """Scan and display local hardware capabilities and memory limits."""
    hw = get_hardware_profile()
    table = Table(title='Hardware Capabilities')
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
    method: QuantMethod = typer.Option(
        QuantMethod.HQQ,
        "--method",
        help="Quantization algorithm (hqq, rtn, ternary, intx, gptq, awq, autoround)",
    ),
    bits: int = typer.Option(2, "--bits", "-b", help="Bit-width (1, 2, 3, 4, 8)"),
    group_size: int = typer.Option(
        128, "--group-size", "-g", help="Group size for groupwise quantization"
    ),
    max_vram: str = typer.Option(
        "3584MB",
        "--max-vram",
        help="Hard VRAM budget ceiling (e.g. '2GB', '1500MB', '4GiB', '2048')",
    ),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to execute on (cuda/cpu)"),
):
    """Milestones 1 & 2: Benchmark single matrix and micro-tiler under strict VRAM caps."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    console.print(
        f"[bold cyan]Benchmarking Matrix Quantization:[/bold cyan] [{out_features} x {in_features}] | Method: {method.value} ({bits}-bit) | User VRAM Cap: {max_vram_mb:.0f} MiB ({max_vram})"
    )

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

    table = Table(
        title="[bold green]Matrix Quantization & Memory Report[/bold green]", show_header=True
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row(
        "Matrix Dimensions",
        f"{out_features} x {in_features} ({out_features * in_features:,} params)",
    )
    table.add_row("Original Size (FP16)", format_size(raw_bytes))
    table.add_row(
        "Quantized Size", f"{format_size(quant_bytes)} ({compression_ratio:.2f}x compression)"
    )
    table.add_row("Effective Bit-width", f"{quant_res.bit_width:.2f} bits/weight")
    table.add_row("User-Specified VRAM Cap", f"{max_vram_mb:.0f} MiB ({max_vram})")
    table.add_row("Measured Peak CUDA VRAM", f"[bold magenta]{peak_vram_mb:.2f} MiB[/bold magenta]")
    table.add_row("Relative Weight Distortion (Frobenius)", f"{report.weight_distortion:.6f}")
    table.add_row("Signal-to-Noise Ratio (SNR)", f"{report.snr_db:.2f} dB")
    table.add_row("Cosine Similarity", f"{report.cosine_similarity:.6f}")

    vram_status = (
        "[bold green]PASS (Strictly Under Budget)[/bold green]"
        if peak_vram_mb <= max_vram_mb
        else "[bold red]FAIL (Exceeded Budget)[/bold red]"
    )
    table.add_row("VRAM Enforcement Status", vram_status)
    console.print(table)


@app.command()
def optimize_precision(
    model: str = typer.Argument(
        ..., help="Model ID (e.g. Qwen/Qwen1.5-MoE-A2.7B) or local directory"
    ),
    target_bpw: float = typer.Option(
        2.2, "--target-bpw", "-b", help="Target average bits-per-weight across model"
    ),
    output_map: str = typer.Option(
        "precision_map.json", "--output", "-o", help="Output path for precision assignment JSON"
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
):
    """Milestone 8: Solve Pareto-optimal heterogeneous precision allocation map."""
    console.print(
        f"[bold cyan]Solving Pareto Precision Map for {model} (Target: {target_bpw:.2f} bpw)...[/bold cyan]"
    )
    table_idx = build_tensor_address_table(model, token=token)
    all_tensors = list(table_idx.tensors.values())

    allocator = ParetoBitAllocator(target_bpw=target_bpw)
    pmap = allocator.solve(model, all_tensors, table_idx.metadata)

    with open(output_map, "w", encoding="utf-8") as f:
        f.write(pmap.model_dump_json(indent=2))

    table = Table(
        title="[bold green]Pareto Precision Optimization Summary[/bold green]", show_header=True
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Target Average Bits/Weight", f"{target_bpw:.2f} bpw")
    table.add_row(
        "Effective Average Bits/Weight",
        f"[bold magenta]{pmap.effective_bpw:.2f} bpw[/bold magenta]",
    )
    table.add_row("Total Tensors Mapped", str(len(pmap.tensor_quant_configs)))
    table.add_row(
        "Estimated Compression Ratio", f"{16.0 / max(0.1, pmap.effective_bpw):.2f}x vs FP16"
    )
    table.add_row("Saved Map To", output_map)
    console.print(table)


@app.command()
def export(
    checkpoint_dir: str = typer.Argument(
        ..., help="Directory containing quantized Safetensors checkpoint"
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Target output file (for GGUF) or directory (for vLLM)"
    ),
    format: str = typer.Option("gguf", "--format", "-f", help="Target format: gguf or vllm"),
):
    """Milestone 9: Export quantized model checkpoint to GGUF (llama.cpp) or vLLM format."""
    console.print(
        f"[bold cyan]Exporting checkpoint {checkpoint_dir} to format: {format.upper()}...[/bold cyan]"
    )

    validator = CheckpointValidator(checkpoint_dir)
    scorecard = validator.validate()
    if not scorecard.is_valid:
        console.print("[bold red]Checkpoint validation failed. Cannot export.[/bold red]")
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
    intermediate_size: int = typer.Option(
        2048,
        "--intermediate-size",
        "-i",
        help="MoE expert intermediate dimension (e.g. 2048 for DeepSeek)",
    ),
    hidden_size: int = typer.Option(
        7168, "--hidden-size", "-k", help="Model hidden dimension (e.g. 7168 for DeepSeek)"
    ),
    method: QuantMethod = typer.Option(
        QuantMethod.HQQ, "--method", "-m", help="Quantization algorithm"
    ),
    bits: int = typer.Option(2, "--bits", "-b", help="Bit-width"),
    group_size: int = typer.Option(128, "--group-size", "-g", help="Group size"),
    max_vram: str = typer.Option(
        "3584MB", "--max-vram", help="Hard VRAM budget ceiling (e.g. '2GB', '1500MB', '4GiB')"
    ),
    device: str = typer.Option("cuda", "--device", "-d", help="Execution device"),
):
    """Milestone 4: Quantize all 3 projection matrices of a single MoE expert under strict VRAM caps."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    console.print(
        f"[bold cyan]Benchmarking Single MoE Expert Quantization:[/bold cyan] Intermediate: {intermediate_size}, Hidden: {hidden_size} | Method: {method.value} ({bits}-bit) | VRAM Cap: {max_vram_mb:.0f} MiB ({max_vram})"
    )

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

    table = Table(
        title="[bold green]Single MoE Expert Quantization Report[/bold green]", show_header=True
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    total_params = (intermediate_size * hidden_size * 2) + (hidden_size * intermediate_size)
    table.add_row("Total Expert Parameters", f"{total_params:,} ({format_params(total_params)})")
    table.add_row("Original Size (FP16)", format_size(total_raw_bytes))
    table.add_row(
        "Quantized Size", f"{format_size(total_quant_bytes)} ({compression_ratio:.2f}x compression)"
    )
    table.add_row("User-Specified VRAM Cap", f"{max_vram_mb:.0f} MiB ({max_vram})")
    table.add_row("Measured Peak CUDA VRAM", f"[bold magenta]{peak_vram_mb:.2f} MiB[/bold magenta]")
    table.add_row("Down Proj Cosine Sim", f"{report.cosine_similarity:.6f}")
    table.add_row("Down Proj SNR", f"{report.snr_db:.2f} dB")

    vram_status = (
        "[bold green]PASS (Strictly Under Budget)[/bold green]"
        if peak_vram_mb <= max_vram_mb
        else "[bold red]FAIL (Exceeded Budget)[/bold red]"
    )
    table.add_row("VRAM Enforcement Status", vram_status)
    console.print(table)


@app.command()
def quantize(
    model: str = typer.Argument(
        ..., help="Model ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0) or local path"
    ),
    output_dir: str = typer.Option(
        "./quantized_model", "--output-dir", "-o", help="Target output folder"
    ),
    method: QuantMethod = typer.Option(
        QuantMethod.HQQ,
        "--method",
        "-m",
        help="Quantization algorithm (hqq, ternary, rtn, intx, gptq, awq, autoround)",
    ),
    bits: int = typer.Option(2, "--bits", "-b", help="Bit-width (1, 2, 3, 4, 8)"),
    group_size: int = typer.Option(
        128, "--group-size", "-g", help="Group size for groupwise quantization"
    ),
    max_vram: str = typer.Option(
        "3584MB", "--max-vram", help="Hard peak VRAM ceiling (e.g. '2GB', '4GB', '1500MB', '2048')"
    ),
    max_cpu_staging: str = typer.Option(
        "2048MB",
        "--max-cpu-staging",
        help="Max staging buffer before shard flush (e.g. '2GB', '1024MB')",
    ),
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
    checkpoint_dir: str = typer.Argument(
        ..., help="PocketTitan package or legacy Safetensors checkpoint"
    ),
    mode: str = typer.Option("fast", "--mode", help="PocketTitan validation mode: fast or full"),
):
    """Validate a .ptitan package, or a legacy Safetensors prototype."""
    console.print(f"[bold cyan]Auditing checkpoint integrity in: {checkpoint_dir}[/bold cyan]")
    if (Path(checkpoint_dir) / "manifest.json").is_file():
        if mode not in {"fast", "full"}:
            console.print("[bold red]--mode must be 'fast' or 'full'[/bold red]")
            raise typer.Exit(code=2)
        report = PtitanValidator(checkpoint_dir).validate(mode=mode)
        table = Table(title="[bold green]PocketTitan Package Integrity[/bold green]")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Mode", report.mode)
        table.add_row("Items checked", f"{report.items_checked:,}")
        table.add_row("Bytes checked", format_size(report.bytes_checked))
        table.add_row("Errors", str(len(report.errors)))
        table.add_row("Warnings", str(len(report.warnings)))
        table.add_row(
            "Status",
            "[bold green]PASS[/bold green]" if report.is_valid else "[bold red]FAIL[/bold red]",
        )
        console.print(table)
        for error in report.errors:
            console.print(f"[bold red]Error:[/bold red] {escape(error)}")
        for warning in report.warnings:
            console.print(f"[bold yellow]Warning:[/bold yellow] {escape(warning)}")
        if not report.is_valid:
            raise typer.Exit(code=1)
        return

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

    status_str = (
        "[bold green]PASS (Valid Checkpoint)[/bold green]"
        if scorecard.is_valid
        else "[bold red]FAIL (Corrupted/Incomplete)[/bold red]"
    )
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
    model: str = typer.Argument(
        ..., help="Hugging Face repo ID (e.g. zai-org/GLM-5.3-Flash) or local directory path"
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    max_vram: str = typer.Option(
        "3584MB", "--max-vram", help="Hard VRAM budget ceiling (e.g. '2GB', '3.5GB', '4GiB')"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output raw JSON instead of formatted tables"
    ),
):
    """Milestone 0: Inspect model repository, architecture, parameter count, shards, and work unit bounds."""
    max_vram_mb = parse_memory_to_mb(max_vram)
    if not json_output:
        console.print(
            f"[bold green]Inspecting model metadata for [cyan]{model}[/cyan] (VRAM Cap: {max_vram_mb:.0f} MiB / {max_vram})...[/bold green]"
        )

    try:
        table_idx = build_tensor_address_table(model, token=token)
        meta = table_idx.metadata
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

    console.print(
        Panel(
            arch_info,
            title=f"[bold green]PocketTitan Inspector: {model}[/bold green]",
            border_style="cyan",
        )
    )

    # Theoretical Storage Footprint Table
    fp16_bytes = meta.total_params * 2
    footprint_table = Table(
        title="[bold]Estimated Model Storage Footprint by Precision[/bold]", show_header=True
    )
    footprint_table.add_column("Precision / Format", style="cyan")
    footprint_table.add_column("Effective bpw", style="yellow")
    footprint_table.add_column("Theoretical Model Size", style="green")
    footprint_table.add_column("PocketTitan Streaming Peak VRAM", style="magenta")

    footprint_table.add_row(
        "FP16 / BF16 (Uncompressed)", "16.0", format_size(fp16_bytes), "N/A (Streamed)"
    )
    footprint_table.add_row(
        "FP8", "8.0", format_size(meta.total_params * 1), f"< {max_vram_mb:.0f} MiB"
    )
    footprint_table.add_row(
        "INT4 / HQQ4", "4.0", format_size(meta.total_params * 0.5), f"< {max_vram_mb:.0f} MiB"
    )
    footprint_table.add_row(
        "INT3 / HQQ3", "3.0", format_size(meta.total_params * 0.375), f"< {max_vram_mb:.0f} MiB"
    )
    footprint_table.add_row(
        "INT2 / HQQ2", "2.0", format_size(meta.total_params * 0.25), f"< {max_vram_mb:.0f} MiB"
    )
    footprint_table.add_row(
        "BitNet / Ternary W1.58",
        "1.58",
        format_size(meta.total_params * 0.20),
        f"< {max_vram_mb:.0f} MiB",
    )
    console.print(footprint_table)

    # Top 5 Largest Tensors & Tiling Sizing
    largest = table_idx.largest_tensors(top_n=5)
    tensor_table = Table(
        title=f"[bold]Largest Tensors & Hardware Work Unit Decomposition (< {max_vram_mb:.0f} MiB Cap)[/bold]",
        show_header=True,
    )
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
            tiling_desc = (
                f"[green]Single Pass[/green] (~{bounds['estimated_vram_per_tile_mb']:.0f}MB VRAM)"
            )
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
    checkpoint_dir: str = typer.Argument(
        ..., help="Path to PocketTitan quantized checkpoint directory"
    ),
    layer: int = typer.Option(0, "--layer", "-l", help="Layer index to inspect"),
):
    """Inspect and verify dequantized layer numerical fidelity and stats."""
    from pockettitan.inference import PocketTitanModelRunner

    runner = PocketTitanModelRunner(checkpoint_dir)
    res = runner.inspect_layer_sample(layer)
    console.print(f"[bold green]Inspection Result:[/bold green] {res}")


@app.command()
def plan(
    model: str = typer.Argument(
        ..., help="Hugging Face repo ID (e.g. Qwen/Qwen3.8-Flash-Next) or local directory path"
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        help="Immutable Hugging Face commit SHA (resolved automatically when omitted)",
    ),
    profile: str = typer.Option("full", "--profile", help="Build profile: canary or full"),
    precision: str = typer.Option(
        "pt-q4e",
        "--precision",
        "-p",
        help="Precision preset: pt-q4e, pt-q2e, bf16, int8, int4, int3, int2, ternary",
    ),
    features: str = typer.Option(
        "text", "--features", help="Comma-separated capabilities to keep: text,vision,mtp"
    ),
    alignment: int = typer.Option(
        4096,
        "--expert-alignment",
        help="Expert record pitch in bytes; page alignment keeps reads page-aligned",
    ),
    workers: int = typer.Option(16, "--workers", help="Parallel shard header requests"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the plan as JSON instead of tables"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the plan JSON to this path"
    ),
):
    """R1: Plan a PocketTitan package — capability filter, expert repacking, precision, and PLE row store.

    Reads only checkpoint headers. No weights are fetched, so the byte-exact
    layout can be reviewed before a multi-hundred-gigabyte build starts.
    """
    profile = profile.strip().lower()
    if profile not in ("canary", "full"):
        console.print(f"[bold red]Invalid --profile '{profile}'. Must be 'canary' or 'full'.[/bold red]")
        raise typer.Exit(code=1)

    try:
        selected = [Capability(f.strip().lower()) for f in features.split(",") if f.strip()]
    except ValueError as e:
        console.print(f"[bold red]Invalid option value:[/bold red] {escape(str(e))}")
        raise typer.Exit(code=1)

    try:
        precision_map = get_precision_preset(precision)
    except KeyError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    if not json_output:
        console.print(
            f"[bold green]Planning package for [cyan]{model}[/cyan][/bold green] "
            f"[dim](profile={profile}, precision={precision_map.name}, features={','.join(c.value for c in selected)})[/dim]"
        )

    try:
        with (
            console.status("[green]Reading shard headers...", spinner="dots")
            if not json_output
            else nullcontext()
        ):
            scan = scan_checkpoint(
                model, token=token, revision=revision, max_workers=workers, strict=True
            )
        build_plan = plan_package(
            scan,
            precision_map=precision_map,
            features=selected,
            build_profile=profile,
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
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        help="Immutable Hugging Face commit SHA (resolved automatically when omitted)",
    ),
    profile: str = typer.Option("full", "--profile", help="Build profile: canary or full"),
    precision: str = typer.Option("pt-q4e", "--precision", "-p", help="Precision preset"),
    features: str = typer.Option("text", "--features", help="Comma-separated capabilities to keep"),
    method: QuantMethod = typer.Option(QuantMethod.RTN, "--method", help="Quantizer backend"),
    max_vram: str = typer.Option("3584MB", "--max-vram", help="Hard VRAM ceiling"),
    device: str = typer.Option("cuda", "--device", "-d", help="Execution device (cuda/cpu)"),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Rebuild from scratch, ignoring the journal"
    ),
    workers: int = typer.Option(16, "--workers", help="Parallel shard header requests"),
    download_workers: int = typer.Option(
        3,
        "--download-workers",
        min=1,
        max=8,
        help="Concurrent remote tensor downloads feeding the single quantizer",
    ),
    max_inflight_source: str = typer.Option(
        "2048MB",
        "--max-inflight-source",
        help="Hard RAM allowance for downloaded/in-flight source tensors",
    ),
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

    budget = MemoryBudgetConfig(
        max_vram_mb=parse_memory_to_mb(max_vram),
        max_cpu_staging_mb=parse_memory_to_mb(max_inflight_source),
    )
    console.print(
        f"[bold green]Packaging [cyan]{model}[/cyan] -> [cyan]{output}[/cyan][/bold green] "
        f"[dim](precision={precision_map.name}, method={method.value}, VRAM cap {budget.max_vram_mb:.0f} MiB)[/dim]"
    )

    try:
        with console.status("[green]Reading shard headers...", spinner="dots"):
            scan = scan_checkpoint(
                model, token=token, revision=revision, max_workers=workers, strict=True
            )
        build_plan = plan_package(
            scan,
            precision_map=precision_map,
            features=selected,
            quant_method=method,
            build_profile=profile,
            complete_model=profile == "full",
            pockettitan_version=__version__,
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
        reader = RemoteTensorSliceReader(model, token=token, revision=scan.source_revision)

    writer = PackageWriter(
        build_plan,
        output,
        reader,
        budget=budget,
        method=method,
        device=device,
        resume=not no_resume,
        download_workers=download_workers,
    )

    from rich.progress import (
        BarColumn,
        Progress,
        ProgressColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    class PipelineDetailColumn(ProgressColumn):
        """Render byte and work-item tasks without pretending items are bytes."""

        def render(self, task):
            total = task.total or 0
            if task.fields.get("unit") == "bytes":
                speed = task.speed or 0.0
                suffix = f" · {format_size(speed)}/s" if speed else ""
                return Text(
                    f"{format_size(task.completed)} / {format_size(total)}{suffix}",
                    style="progress.data.speed",
                )
            return Text(
                f"{int(task.completed):,} / {int(total):,} items",
                style="progress.data.speed",
            )

    console.print()
    try:
        resume_status = writer.resume_status()
        if resume_status.items_completed:
            console.print(
                "[bold cyan]Resuming:[/bold cyan] "
                f"{resume_status.items_completed:,}/{resume_status.total_items:,} work items "
                f"already committed; skipping "
                f"{format_size(resume_status.source_bytes_completed)} of source reads."
            )
        source_is_remote = not scan.is_local
        active_download_workers = download_workers if source_is_remote else 1
        console.print(
            "[dim]Pipeline: "
            f"{active_download_workers} downloader(s) -> 1 {writer.device.upper()} quantizer "
            f"-> 1 durable writer · source staging <= "
            f"{format_size(budget.max_cpu_staging_mb * 1024 * 1024)}[/dim]"
        )
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            PipelineDetailColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            download_task = (
                progress.add_task(
                    f"Downloading ({active_download_workers} workers)",
                    total=build_plan.source_read_bytes,
                    completed=resume_status.source_bytes_completed,
                    unit="bytes",
                )
                if source_is_remote
                else None
            )
            process_task = progress.add_task(
                "Waiting for source tensors...",
                total=build_plan.num_work_items,
                completed=resume_status.items_completed,
                unit="items",
            )

            def advance_download(delta):
                if download_task is not None:
                    progress.update(download_task, advance=delta)

            result = writer.build(
                on_start=lambda region, label: progress.update(
                    process_task,
                    description=f"Quantizing [bold green]{label}[/bold green]",
                ),
                on_item=lambda region, label: progress.update(process_task, advance=1),
                on_bytes=advance_download if source_is_remote else None,
            )
    except Exception as e:
        console.print(f"[bold red]Build failed:[/bold red] {escape(str(e))}")
        console.print("[dim]Rerun the same command to resume from the journal.[/dim]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]Package written to {output}[/bold green]\n"
        f"  items written {result.items_written:,} · skipped {result.items_skipped:,}\n"
        f"  bytes written {format_size(result.bytes_written)}\n"
        f"  download workers {result.download_workers} · peak source staging "
        f"{format_size(result.peak_inflight_source_bytes)}\n"
        f"  peak VRAM {result.peak_vram_mb:.1f} MiB / {budget.usable_vram_mb:.0f} MiB usable\n"
        f"  elapsed {result.elapsed_s:.1f} s"
    )


@app.command()
def audit(
    model: str = typer.Argument(
        ..., help="Hugging Face repo ID (e.g. Qwen/Qwen3.8-Flash-Next) or local directory path"
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face API token"),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        help="Immutable Hugging Face commit SHA (resolved automatically when omitted)",
    ),
    precision: str = typer.Option(
        "pt-q4e",
        "--precision",
        "-p",
        help="Precision preset: pt-q4e, pt-q2e, bf16, int8, int4, int3, int2, ternary",
    ),
    features: str = typer.Option(
        "text", "--features", help="Comma-separated capabilities to keep: text,vision,mtp"
    ),
    ram_budget: str = typer.Option(
        "7GB",
        "--ram-budget",
        help="RAM available for the expert cache (drives roofline slot count)",
    ),
    workers: int = typer.Option(16, "--workers", help="Parallel shard header requests"),
    no_strict: bool = typer.Option(
        False, "--no-strict", help="Continue past unreadable shards instead of failing"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the report as JSON instead of tables"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the report JSON to this path"
    ),
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
        with (
            console.status("[green]Reading shard headers...", spinner="dots")
            if not json_output
            else nullcontext()
        ):
            scan = scan_checkpoint(
                model,
                token=token,
                revision=revision,
                max_workers=workers,
                strict=not no_strict,
            )
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


@app.command()
def sim(
    tokens: int = typer.Option(500, "--tokens", "-n", help="Number of decoding steps to simulate"),
    distribution: str = typer.Option("zipf", "--distribution", "-d", help="Synthetic trace distribution: zipf, uniform, sticky"),
    alpha: float = typer.Option(1.0, "--alpha", "-a", help="Zipf distribution skew parameter alpha"),
    bits: float = typer.Option(4.0, "--bits", "-b", help="Nominal bits per weight (4.0 or 2.0)"),
    ssd_bw: float = typer.Option(3.5, "--ssd-bw", help="SSD bandwidth in GB/s"),
    capacities: str = typer.Option("512,1024,2048,2880,4096,5437", "--capacities", help="Comma-separated cache capacities in expert slots"),
    trace_file: Optional[Path] = typer.Option(None, "--trace", "-t", help="Path to real routing trace file (.jsonl / .jsonl.gz)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of formatted tables"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write report JSON to file"),
):
    """R2: Trace Simulator — replay expert routing through cache policies and model roofline throughput."""
    from pockettitan.sim import (
        DistributionType,
        HardwareProfile,
        generate_synthetic_trace,
        render_simulation_report,
        run_capacity_sweep,
    )
    from pockettitan.profiler.trace import TraceReader

    try:
        cap_list = [int(c.strip()) for c in capacities.split(",") if c.strip()]
    except ValueError:
        console.print(f"[bold red]Invalid --capacities list '{capacities}'. Must be integers.[/bold red]")
        raise typer.Exit(code=1)

    hw = HardwareProfile(ssd_bandwidth_gbps=ssd_bw)

    if trace_file is not None:
        if not trace_file.exists():
            console.print(f"[bold red]Trace file not found:[/bold red] {trace_file}")
            raise typer.Exit(code=1)
        if not json_output:
            console.print(f"[bold green]Replaying Real MoE Trace:[/bold green] [cyan]{trace_file}[/cyan]")
        trace = TraceReader(trace_file).read_all_events()
    else:
        try:
            dist_enum = DistributionType(distribution.lower())
        except ValueError:
            console.print(f"[bold red]Invalid distribution '{distribution}'.[/bold red] Choose from: zipf, uniform, sticky")
            raise typer.Exit(code=1)

        if not json_output:
            console.print(
                f"[bold green]Simulating Synthetic MoE Trace[/bold green] "
                f"[dim]({tokens} tokens, dist={dist_enum.value}, alpha={alpha}, bits={bits}b, ssd={ssd_bw} GB/s)[/dim]"
            )

        trace = generate_synthetic_trace(
            num_tokens=tokens,
            num_layers=48,
            num_experts=512,
            top_k=10,
            distribution=dist_enum,
            alpha=alpha,
        )

    report = run_capacity_sweep(
        events=trace,
        capacities=cap_list,
        bits_per_weight=bits,
        hardware=hw,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if json_output:
        console.print_json(report.model_dump_json())
    else:
        render_simulation_report(console, report)
        if output is not None:
            console.print(f"\n[dim]Simulation report written to {output}[/dim]")


@app.command()
def gate(
    trace_file: Optional[Path] = typer.Option(None, "--trace", "-t", help="Path to real routing trace file (.jsonl / .jsonl.gz)"),
    tokens: int = typer.Option(500, "--tokens", "-n", help="Synthetic tokens if no trace file provided"),
    alpha: float = typer.Option(1.0, "--alpha", "-a", help="Zipf distribution skew alpha"),
    slots: int = typer.Option(2880, "--slots", "-s", help="Target RAM cache budget in expert slots (default 2880 = 7.0 GB RAM)"),
    bits: float = typer.Option(4.0, "--bits", "-b", help="Bits per weight (4.0 or 2.0)"),
    ssd_bw: float = typer.Option(3.5, "--ssd-bw", help="SSD bandwidth in GB/s"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write formal gate report markdown to file"),
):
    """R4: The Oracle Decision Gate — evaluate feasibility threshold (50% Oracle hit rate @ 2,880 slots)."""
    from pockettitan.sim import (
        DistributionType,
        HardwareProfile,
        evaluate_oracle_gate,
        format_gate_report_markdown,
        generate_synthetic_trace,
    )
    from pockettitan.profiler.trace import TraceReader

    hw = HardwareProfile(ssd_bandwidth_gbps=ssd_bw)

    if trace_file is not None:
        if not trace_file.exists():
            console.print(f"[bold red]Trace file not found:[/bold red] {trace_file}")
            raise typer.Exit(code=1)
        console.print(f"[bold green]Running R4 Oracle Gate on Real Trace:[/bold green] [cyan]{trace_file}[/cyan]")
        events = TraceReader(trace_file).read_all_events()
    else:
        console.print(
            f"[bold green]Running R4 Oracle Gate on Synthetic Trace[/bold green] "
            f"[dim]({tokens} tokens, alpha={alpha}, target_slots={slots}, bits={bits}b, ssd={ssd_bw} GB/s)[/dim]"
        )
        events = generate_synthetic_trace(
            num_tokens=tokens,
            num_layers=48,
            num_experts=512,
            top_k=10,
            distribution=DistributionType.ZIPF,
            alpha=alpha,
        )

    report = evaluate_oracle_gate(
        events=events,
        target_slots=slots,
        bits_per_weight=bits,
        hardware=hw,
    )

    # Render results table
    table = Table(title="[bold green]Phase R4 — The Oracle Decision Gate[/bold green]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Target Budget Slots", f"{report.target_budget_slots:,} slots (~7.0 GB RAM)")
    table.add_row("Oracle Hit Rate (Upper Bound)", f"[bold]{report.oracle_hit_rate_at_budget * 100:.1f}%[/bold]")
    table.add_row("OS Page Cache Hit Rate", f"{report.os_page_cache_hit_rate * 100:.1f}%")
    table.add_row("Winning Online Policy", f"{report.winning_online_policy} (+{report.custom_policy_advantage * 100:.1f}%)")
    table.add_row("Expert Gini Skew", f"{report.gini_coefficient:.4f}")
    
    decision_style = "bold green" if report.decision.value != "KILL_CUSTOM_CACHE" else "bold red"
    table.add_row("Gate Decision", f"[{decision_style}]{report.decision.value}[/{decision_style}]")

    console.print(table)
    console.print(f"\n[bold]{report.rationale}[/bold]\n")

    md_report = format_gate_report_markdown(report)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md_report, encoding="utf-8")
        console.print(f"[dim]Formal gate report written to {output}[/dim]")


profile_app = typer.Typer(
    help="R3: MoE Routing Profiler — generate prompt suites and analyze routing traces."
)
app.add_typer(profile_app, name="profile")


@profile_app.command("prompts")
def profile_prompts(
    samples_per_task: int = typer.Option(5, "--samples", "-n", help="Number of prompt samples per task category"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File to write JSON benchmark suite to"),
):
    """Generate standardized benchmark prompts across the 5 canonical tasks."""
    from pockettitan.profiler.prompts import generate_benchmark_suite

    prompts = generate_benchmark_suite(num_samples_per_task=samples_per_task)
    data = [p.model_dump() for p in prompts]

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[bold green]Generated {len(prompts)} benchmark prompts -> {output}[/bold green]")
    else:
        table = Table(title="[bold green]PocketTitan Benchmark Prompts (R3)[/bold green]", show_header=True)
        table.add_column("Prompt ID", style="cyan")
        table.add_column("Task Category", style="magenta")
        table.add_column("User Prompt Excerpt", style="white")

        for p in prompts:
            excerpt = p.user_prompt[:80] + "..." if len(p.user_prompt) > 80 else p.user_prompt
            table.add_row(p.prompt_id, p.task_type.value, excerpt)

        console.print(table)


@profile_app.command("analyze")
def profile_analyze(
    trace_file: Path = typer.Argument(..., help="Path to routing trace file (.jsonl / .jsonl.gz)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write metrics JSON to file"),
):
    """Compute comprehensive Gini skew, entropy, and cross-layer overlap metrics from a trace file."""
    from pockettitan.profiler.trace import TraceReader, analyze_routing_trace

    if not trace_file.exists():
        console.print(f"[bold red]Trace file not found:[/bold red] {trace_file}")
        raise typer.Exit(code=1)

    events = TraceReader(trace_file).read_all_events()
    metrics = analyze_routing_trace(events)

    table = Table(title=f"[bold green]MoE Routing Profile: {trace_file.name}[/bold green]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tokens", f"{metrics.total_tokens:,}")
    table.add_row("Total Routing Events", f"{metrics.total_events:,}")
    table.add_row("Unique Experts Activated", f"{metrics.unique_experts_accessed:,}")
    table.add_row("Expert Gini Coefficient (Skew)", f"{metrics.gini_coefficient:.4f}")
    table.add_row("Mean Router Entropy", f"{metrics.mean_router_entropy:.4f}")
    table.add_row("Cross-Layer Top-K Overlap", f"{metrics.cross_layer_overlap_mean * 100:.2f}%")

    console.print(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\n[dim]Analysis written to {output}[/dim]")


@app.command()
def run(
    package: Path = typer.Argument(..., help='Path to a .ptitan package directory'),
    prompt: str = typer.Option('Explain what a memory hierarchy is.', '--prompt', '-p'),
    max_new_tokens: int = typer.Option(64, '--max-new-tokens', '-n'),
    temperature: float = typer.Option(0.0, '--temperature', '-t', help='0 = greedy'),
    device: str = typer.Option('auto', '--device', help='cuda, cpu, or auto'),
    dtype: str = typer.Option('float32', '--dtype', help='float32, float16, bfloat16'),
    cache_mb: float = typer.Option(512.0, '--cache-mb', help='Decoded-weight cache ceiling'),
    chat: bool = typer.Option(True, '--chat/--raw', help='Apply the tokenizer chat template'),
):
    """Generate text from a .ptitan package through the reference HF module tree.

    Weights stay on disk and are decoded per use, so a package larger than RAM
    still runs - slowly. This is a correctness tool, not the fast path.
    """
    from pockettitan.runtime.hf import generate as run_generation

    def announce(info):
        console.print(
            '  [green]' + str(info['backed_by_package']) + '[/green] tensors from the package, '
            + '[green]' + str(info['materialized']) + '[/green] materialized, '
            + format(info['resident_params'], ',') + ' resident params'
        )

    console.print('[bold cyan]Loading[/bold cyan] ' + str(package))
    try:
        result = run_generation(
            package,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_bytes=int(cache_mb * 1024 * 1024),
            chat=chat,
            on_load=announce,
        )
    except FileNotFoundError as exc:
        console.print('[bold red]' + escape(str(exc)) + '[/bold red]')
        raise typer.Exit(code=1)

    console.print()
    console.print('[bold yellow]Prompt[/bold yellow] ' + escape(result.prompt))
    console.print('[bold green]Output[/bold green] ' + escape(result.text))
    console.print()
    for line in result.summary_lines():
        console.print('[dim]' + line + '[/dim]')


if __name__ == "__main__":
    app()
