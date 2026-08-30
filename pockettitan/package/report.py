"""Rich rendering for build plans (R1)."""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pockettitan.audit.report import fmt_bytes, fmt_params
from pockettitan.package.plan import BuildPlan


def render_plan_summary(plan: BuildPlan) -> Panel:
    manifest = plan.manifest
    totals = manifest.totals

    lines = [
        f"[bold]Source[/bold]           {manifest.source_model}",
        f"[bold]Architecture[/bold]     {manifest.architecture or 'unknown'}",
        f"[bold]Features[/bold]         {', '.join(manifest.features)}",
        f"[bold]Precision map[/bold]    {manifest.precision_map_name}",
        "",
        f"[bold]Packaged params[/bold]  [bold green]{totals.packaged_params:,}[/bold green]  ({fmt_params(totals.packaged_params)})",
        f"[bold]Dropped params[/bold]   {totals.dropped_params:,}  ({fmt_params(totals.dropped_params)})",
        f"[bold]Package size[/bold]     [bold cyan]{fmt_bytes(totals.total_bytes)}[/bold cyan]  ({totals.average_bits:.2f} bits/param)",
        f"[bold]Source to read[/bold]   {fmt_bytes(plan.source_read_bytes)}",
        f"[bold]Work items[/bold]       {plan.num_work_items:,}",
        "",
        f"[bold]Activated/token[/bold]  {manifest.activated_params_per_token:,}  ({fmt_params(manifest.activated_params_per_token)})",
        f"[bold]Expert I/O/token[/bold] [bold magenta]{fmt_bytes(manifest.expert_bytes_per_token)}[/bold magenta] across {manifest.reads_per_token} reads",
    ]
    return Panel(
        "\n".join(lines),
        title="[bold green]PocketTitan Build Plan - R1[/bold green]",
        border_style="green",
    )


def render_regions(plan: BuildPlan) -> Table:
    table = Table(title="[bold]Package Regions[/bold]", show_header=True, header_style="bold")
    table.add_column("Region", style="cyan")
    table.add_column("Contents")
    table.add_column("Items", justify="right")
    table.add_column("Bytes", justify="right", style="white")

    totals = plan.manifest.totals
    table.add_row("dense/", "VRAM-resident core", f"{len(plan.dense):,}", fmt_bytes(totals.dense_bytes))

    layout = plan.manifest.expert_layout
    if layout is not None:
        table.add_row(
            "experts/",
            f"{layout.num_experts} experts x {len(layout.layers)} layers, "
            f"{fmt_bytes(layout.record.payload_bytes)}/record",
            f"{layout.num_records:,}",
            fmt_bytes(totals.expert_bytes),
        )
    if plan.ple is not None:
        table.add_row(
            "ple/",
            f"{plan.ple.total_rows:,} rows x {plan.ple.row.payload_bytes} B "
            f"({plan.ple.row.rows_per_page}/page, {plan.ple.row.waste_fraction * 100:.1f}% waste)",
            f"{len(plan.ple.shards):,}",
            fmt_bytes(totals.ple_bytes),
        )

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", "", f"[bold]{plan.num_work_items:,}[/bold]", f"[bold]{fmt_bytes(totals.total_bytes)}[/bold]")
    return table


def render_expert_record(plan: BuildPlan) -> Optional[Table]:
    layout = plan.manifest.expert_layout
    if layout is None:
        return None

    table = Table(
        title="[bold]Expert Record Layout[/bold]  [dim](one expert = one contiguous read)[/dim]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Projection", style="cyan")
    table.add_column("Shape")
    table.add_column("Bits", justify="right")
    table.add_column("Offset", justify="right")
    table.add_column("Length", justify="right")
    table.add_column("Sections", style="dim")

    for projection in layout.record.projections:
        table.add_row(
            projection.name,
            "x".join(str(d) for d in projection.shape),
            f"{projection.bits:g}",
            f"{projection.offset:,}",
            fmt_bytes(projection.length),
            " + ".join(f"{s.section.value}:{s.length:,}" for s in projection.spans),
        )

    record = layout.record
    table.add_section()
    table.add_row(
        "[bold]RECORD[/bold]", f"{record.num_params:,} params", "",
        "0", f"[bold]{fmt_bytes(record.payload_bytes)}[/bold]",
        f"stride {record.stride:,} (+{record.padding_bytes:,} pad, {record.alignment}B aligned)",
    )
    return table


def render_plan(console: Console, plan: BuildPlan) -> None:
    """Print the full build plan."""
    console.print(render_plan_summary(plan))
    console.print()
    console.print(render_regions(plan))
    record_table = render_expert_record(plan)
    if record_table is not None:
        console.print()
        console.print(record_table)
