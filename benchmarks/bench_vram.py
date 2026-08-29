"""PocketTitan VRAM Benchmark & Throughput Profiler."""

import time
from typing import List, Tuple
import torch
from rich.console import Console
from rich.table import Table

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.quantizers import get_quantizer
from pockettitan.scheduler.tiler import MatrixTiler

console = Console()


def run_benchmark():
    matrix_sizes = [
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
    ]
    methods = [
        (QuantMethod.RTN, 4),
        (QuantMethod.TERNARY, 2),
        (QuantMethod.HQQ, 2),
        (QuantMethod.AWQ, 4),
        (QuantMethod.GPTQ, 4),
    ]
    budget = MemoryBudgetConfig(max_vram_mb=3584.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiler = MatrixTiler(budget)

    table = Table(title="[bold green]PocketTitan Hardware VRAM & Throughput Benchmark[/bold green]", show_header=True)
    table.add_column("Matrix Shape", style="cyan")
    table.add_column("Params", style="white")
    table.add_column("Method", style="yellow")
    table.add_column("Bits", style="magenta")
    table.add_column("Peak CUDA VRAM", style="green")
    table.add_column("Time", style="blue")
    table.add_column("Throughput", style="white")

    for shape in matrix_sizes:
        m, k = shape
        w = torch.randn(m, k, dtype=torch.float16, device="cpu") * 0.02
        raw_mb = w.nbytes / (1024 * 1024)
        
        for method, bits in methods:
            cfg = QuantConfig(method=method, bits=bits, group_size=128, device=device)
            quantizer = get_quantizer(cfg)
            
            start_t = time.perf_counter()
            res, peak_vram = tiler.quantize_matrix(w, quantizer=quantizer, target_device=device)
            elapsed = time.perf_counter() - start_t
            
            throughput = raw_mb / max(0.001, elapsed)
            
            table.add_row(
                f"{m} x {k}",
                f"{(m * k) / 1e6:.1f}M",
                method.value.upper(),
                f"{bits}-bit",
                f"{peak_vram:.1f} MiB" if device == "cuda" else "CPU",
                f"{elapsed:.2f}s",
                f"{throughput:.1f} MB/s",
            )
            del res
            
    console.print(table)


if __name__ == "__main__":
    run_benchmark()
