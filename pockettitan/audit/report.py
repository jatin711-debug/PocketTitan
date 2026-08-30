"""Rich rendering for audit reports (R0).

Kept separate from the CLI so the analysis can be rendered from notebooks and
tests without importing Typer.

Glyphs are resolved against the destination stream's encoding. Windows consoles
still default to cp1252, and a report that raises ``UnicodeEncodeError`` halfway
through rendering is worse than one that prints ``OK`` instead of a check mark.
"""

from typing import Dict, List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pockettitan.audit.budget import AuditReport
from pockettitan.audit.classify import Tier

DEFAULT_CONTEXTS: Sequence[int] = (2048, 4096, 8192, 32768, 131072)

UNICODE_GLYPHS: Dict[str, str] = {
    "ok": "✓",       # ✓
    "warn": "⚠",     # ⚠
    "bullet": "·",   # ·
    "arrow": "→",    # →
    "inf": "∞",      # ∞
    "times": "×",    # ×
}
ASCII_GLYPHS: Dict[str, str] = {
    "ok": "OK",
    "warn": "!",
    "bullet": "-",
    "arrow": "->",
    "inf": "inf",
    "times": "x",
}

_TIER_STYLE = {
    Tier.VRAM_HOT: "bold red",
    Tier.RAM_WARM: "yellow",
    Tier.NVME_COLD: "cyan",
}
_TIER_LABEL = {
    Tier.VRAM_HOT: "VRAM",
    Tier.RAM_WARM: "RAM",
    Tier.NVME_COLD: "NVMe",
}


def glyphs_for(console: Console) -> Dict[str, str]:
    """Pick the richest glyph set the console's encoding can actually emit."""
    encoding = getattr(getattr(console, "file", None), "encoding", None) or "utf-8"
    try:
        "".join(UNICODE_GLYPHS.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return ASCII_GLYPHS
    return UNICODE_GLYPHS


def _g(glyphs: Optional[Dict[str, str]]) -> Dict[str, str]:
    return glyphs if glyphs is not None else UNICODE_GLYPHS


def fmt_bytes(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"


def fmt_params(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.3f} B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f} M"
    if count >= 1_000:
        return f"{count / 1_000:.2f} K"
    return str(count)


def _tier_cell(tier: Tier) -> str:
    style = _TIER_STYLE[tier]
    return f"[{style}]{_TIER_LABEL[tier]}[/{style}]"


def render_summary(report: AuditReport, glyphs: Optional[Dict[str, str]] = None) -> Panel:
    g = _g(glyphs)
    dtypes = ", ".join(f"{k}{g['times']}{v}" for k, v in report.dtype_histogram.items())

    lines: List[str] = [
        f"[bold]Model[/bold]            {report.model_id}",
        f"[bold]Tensors[/bold]          {report.num_tensors:,} across {report.num_shards} shards",
        f"[bold]Total params[/bold]     [bold cyan]{report.total_params:,}[/bold cyan]  ({fmt_params(report.total_params)})",
        f"[bold]Source bytes[/bold]     {fmt_bytes(report.total_source_bytes)}",
        f"[bold]Dtypes[/bold]           {dtypes}",
        f"[bold]Features[/bold]         {', '.join(c.value for c in report.activation.features)}",
        f"[bold]Enabled params[/bold]   [bold green]{report.enabled_params:,}[/bold green]  ({fmt_params(report.enabled_params)})",
        f"[bold]LM core[/bold]          {report.lm_core_params:,}  ({fmt_params(report.lm_core_params)}) [dim]excl. cold lookup tables[/dim]",
        (
            f"[bold]Activated/token[/bold]  [bold magenta]{report.activation.total:,}[/bold magenta]  "
            f"({fmt_params(report.activation.total)}, "
            f"{100.0 * report.activation.total / max(1, report.total_params):.2f}% of model)"
        ),
        f"[bold]Scan time[/bold]        {report.elapsed_s:.2f} s",
        "",
    ]

    if report.discrepancies:
        lines.append(f"[bold yellow]{g['warn']} {len(report.discrepancies)} discrepancies[/bold yellow]")
        for item in report.discrepancies[:5]:
            lines.append(f"  [yellow]{g['bullet']} {item}[/yellow]")
    else:
        lines.append(
            f"[bold green]{g['ok']} Verified against published index (tensor set + total_size)[/bold green]"
        )

    return Panel(
        "\n".join(lines),
        title="[bold green]PocketTitan Audit - R0[/bold green]",
        border_style="green",
    )


def render_components(report: AuditReport, glyphs: Optional[Dict[str, str]] = None) -> Table:
    table = Table(title="[bold]Component Decomposition[/bold]", show_header=True, header_style="bold")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Cap", style="dim")
    table.add_column("Tier", no_wrap=True)
    table.add_column("Tensors", justify="right")
    table.add_column("Parameters", justify="right", style="white")
    table.add_column("Share", justify="right")
    table.add_column("Source", justify="right", style="dim")

    total = report.total_params
    for stat in report.breakdown.ordered():
        if stat.params == 0:
            continue
        table.add_row(
            stat.component.value,
            stat.capability.value,
            _tier_cell(stat.tier),
            f"{stat.num_tensors:,}",
            f"{stat.params:,}",
            f"{100.0 * stat.share_of(total):.2f}%",
            fmt_bytes(stat.bytes_source),
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]", "", "",
        f"[bold]{report.num_tensors:,}[/bold]",
        f"[bold]{total:,}[/bold]",
        "[bold]100.00%[/bold]",
        f"[bold]{fmt_bytes(report.total_source_bytes)}[/bold]",
    )
    return table


def render_capability(report: AuditReport, glyphs: Optional[Dict[str, str]] = None) -> Table:
    table = Table(title="[bold]Capability Stripping[/bold]", show_header=True, header_style="bold")
    table.add_column("Dropped component", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Source bytes", justify="right")
    table.add_column("Packed @ map", justify="right")
    table.add_column("% of model", justify="right")

    dropped = report.dropped_params
    if not dropped:
        table.add_row("[dim]nothing dropped[/dim]", "-", "-", "-", "-")
        return table

    total_params = total_source = total_packed = 0
    for component, params in sorted(dropped.items(), key=lambda kv: -kv[1]):
        stat = report.breakdown.stats[component]
        packed = int(params * report.precision_map.bits_for(component) / 8)
        total_params += params
        total_source += stat.bytes_source
        total_packed += packed
        table.add_row(
            component.value,
            f"{params:,}",
            fmt_bytes(stat.bytes_source),
            fmt_bytes(packed),
            f"{100.0 * params / max(1, report.total_params):.2f}%",
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL SAVED[/bold]",
        f"[bold]{total_params:,}[/bold]",
        f"[bold]{fmt_bytes(total_source)}[/bold]",
        f"[bold]{fmt_bytes(total_packed)}[/bold]",
        f"[bold]{100.0 * total_params / max(1, report.total_params):.2f}%[/bold]",
    )
    return table


def render_storage(report: AuditReport, glyphs: Optional[Dict[str, str]] = None) -> Table:
    g = _g(glyphs)
    storage = report.storage
    table = Table(
        title=f"[bold]Storage Budget - precision map '{storage.precision_map_name}'[/bold]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Component", style="cyan")
    table.add_column("Tier", no_wrap=True)
    table.add_column("Parameters", justify="right")
    table.add_column("Eff. bits", justify="right")
    table.add_column("Packed", justify="right", style="white")

    for entry in storage.entries:
        table.add_row(
            entry.component.value,
            _tier_cell(entry.tier),
            f"{entry.params:,}",
            f"{entry.effective_bits:.2f}",
            fmt_bytes(entry.packed_bytes),
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL ON NVMe[/bold]", "",
        f"[bold]{storage.total_params:,}[/bold]",
        f"[bold]{storage.average_bits:.2f}[/bold]",
        f"[bold]{fmt_bytes(storage.total_packed_bytes)}[/bold]",
    )
    for tier in (Tier.VRAM_HOT, Tier.RAM_WARM, Tier.NVME_COLD):
        resident = storage.bytes_in_tier(tier)
        if resident:
            style = _TIER_STYLE[tier]
            table.add_row(
                f"[dim]  {g['arrow']} resident in {_TIER_LABEL[tier]}[/dim]", "", "", "",
                f"[{style}]{fmt_bytes(resident)}[/{style}]",
            )
    return table


def render_state(
    report: AuditReport,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    glyphs: Optional[Dict[str, str]] = None,
) -> Table:
    g = _g(glyphs)
    state = report.state
    table = Table(
        title=(
            f"[bold]State Budget[/bold]  "
            f"[dim]({state.num_full_attn_layers} full-attn, {state.num_linear_attn_layers} linear-attn layers)[/dim]"
        ),
        show_header=True,
        header_style="bold",
    )
    table.add_column("Context", justify="right", style="cyan")
    table.add_column("KV + indexer", justify="right")
    table.add_column("Recurrent state", justify="right")
    table.add_column("Total", justify="right", style="white")

    for ctx in contexts:
        table.add_row(
            f"{ctx:,}",
            fmt_bytes(state.bytes_per_token * ctx),
            fmt_bytes(state.recurrent_state_bytes),
            fmt_bytes(state.at_context(ctx)),
        )

    table.caption = (
        f"{state.bytes_per_token / 1024.0:.1f} KiB/token {g['bullet']} "
        f"recurrent state is constant in context length"
    )
    return table


def render_roofline(report: AuditReport, glyphs: Optional[Dict[str, str]] = None) -> Table:
    g = _g(glyphs)
    roofline = report.roofline
    if roofline is None:
        return Table(title="[dim]No MoE routing detected - roofline not applicable[/dim]")

    table = Table(
        title=f"[bold]SSD Roofline - experts @ {roofline.expert_bits:.2f} bits[/bold]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Cache hit", justify="right", style="cyan")
    table.add_column("SSD / token", justify="right")
    bandwidths = sorted({bw for row in roofline.rows for bw in row.tokens_per_second})
    for bw in bandwidths:
        table.add_column(f"{bw:g} GB/s", justify="right")

    for row in roofline.rows:
        cells = [f"{row.hit_rate * 100:.0f}%", fmt_bytes(row.ssd_bytes_per_token)]
        for bw in bandwidths:
            tps = row.tokens_per_second.get(bw, 0.0)
            cells.append(g["inf"] if tps == float("inf") else f"{tps:.1f} tok/s")
        table.add_row(*cells)

    table.caption = (
        f"expert record {fmt_bytes(roofline.expert_record_bytes)} {g['bullet']} "
        f"{roofline.reads_per_token} reads/token {g['bullet']} "
        f"cache {roofline.cache_slots:,}/{roofline.total_expert_slots:,} slots "
        f"({roofline.cache_capacity_fraction * 100:.1f}%) {g['bullet']} "
        f"ceilings only - comparable systems reach ~{roofline.efficiency_factor * 100:.0f}%"
    )
    return table


def render_report(
    console: Console,
    report: AuditReport,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    show_roofline: bool = True,
) -> None:
    """Print the full audit to the console."""
    g = glyphs_for(console)

    console.print(render_summary(report, g))
    console.print()
    console.print(render_components(report, g))
    console.print()
    console.print(render_capability(report, g))
    console.print()
    console.print(render_storage(report, g))
    console.print()
    console.print(render_state(report, contexts, g))
    if show_roofline and report.roofline is not None:
        console.print()
        console.print(render_roofline(report, g))

    if report.breakdown.unclassified:
        console.print()
        console.print(
            f"[yellow]{g['warn']} {len(report.breakdown.unclassified)} unclassified tensors "
            f"(review before trusting this architecture):[/yellow]"
        )
        for name in report.breakdown.unclassified[:10]:
            console.print(f"  [dim]{g['bullet']} {name}[/dim]")
