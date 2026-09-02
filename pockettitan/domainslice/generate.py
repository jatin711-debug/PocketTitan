"""End-to-end multi-token prompt generation backed by DomainSlice paged weights."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

import torch
import torch.nn.functional as F

from pockettitan.domainslice.store import CompositeWeightStore, RemoteHuggingFaceStore
from pockettitan.domainslice.types import ProgressCallback
from pockettitan.runtime.hf.olmoe_model import PagedOlmoeOneTokenRunner, SequentialPassMetrics


def _format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"


@dataclass
class DomainSliceGenerateResult:
    """Detailed telemetry and generated text from a DomainSlice prompt run."""

    prompt: str
    formatted_prompt: str
    prompt_tokens: List[int]
    generated_tokens: List[int]
    generated_text: str
    prefill_elapsed_s: float = 0.0
    decode_elapsed_s: float = 0.0
    total_elapsed_s: float = 0.0
    prefill_tok_per_s: float = 0.0
    decode_tok_per_s: float = 0.0
    overall_tok_per_s: float = 0.0
    global_page_hits: int = 0
    global_page_faults: int = 0
    global_remote_bytes: int = 0
    backbone_page_hits: int = 0
    backbone_page_faults: int = 0
    backbone_remote_bytes: int = 0
    expert_page_hits: int = 0
    expert_page_faults: int = 0
    expert_remote_bytes: int = 0
    experts_executed: int = 0
    total_remote_bytes: int = 0
    total_page_bytes: int = 0
    peak_projection_bytes: int = 0
    peak_head_chunk_bytes: int = 0
    peak_cuda_bytes: int = 0
    peak_rss_bytes: int = 0
    final_kv_cache_bytes: int = 0
    vram_cache_hits: int = 0
    ram_cache_hits: int = 0

    def summary_lines(self) -> List[str]:
        lines = [
            f"Prompt length: {len(self.prompt_tokens)} tokens · Generated: {len(self.generated_tokens)} tokens",
            f"Prefill time: {self.prefill_elapsed_s:.2f}s ({self.prefill_tok_per_s:.2f} tok/s) · "
            f"Decode time: {self.decode_elapsed_s:.2f}s ({self.decode_tok_per_s:.2f} tok/s)",
            f"Total wall time: {self.total_elapsed_s:.2f}s · Overall throughput: {self.overall_tok_per_s:.2f} tok/s",
            f"Remote payload: {_format_bytes(self.total_remote_bytes)} fetched · Logical page access: {_format_bytes(self.total_page_bytes)}",
            f"Page faults: {self.global_page_faults + self.backbone_page_faults + self.expert_page_faults} · "
            f"Page hits: {self.global_page_hits + self.backbone_page_hits + self.expert_page_hits}",
        ]
        if self.vram_cache_hits > 0 or self.ram_cache_hits > 0:
            lines.append(
                f"In-memory hits: {self.vram_cache_hits} VRAM (Tier 1) · {self.ram_cache_hits} Host RAM (Tier 2)"
            )
        lines.extend([
            f"Experts executed: {self.experts_executed:,}",
            f"Peak CUDA allocation: {_format_bytes(self.peak_cuda_bytes)} · Peak process RSS: {_format_bytes(self.peak_rss_bytes)}",
            f"KV cache footprint: {_format_bytes(self.final_kv_cache_bytes)}",
        ])
        return lines


def _sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """Sample next token from final logits with optional temperature and nucleus filtering."""
    flat = logits.float().reshape(-1)
    if temperature <= 0.0:
        return int(flat.argmax().item())

    scaled = flat / float(temperature)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        sorted_logits[sorted_indices_to_remove] = -float("Inf")
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = int(torch.multinomial(probs, 1).item())
        return int(sorted_indices[sampled_idx].item())

    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_olmoe_text(
    store: CompositeWeightStore,
    remote: RemoteHuggingFaceStore,
    *,
    prompt: str,
    tokenizer=None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    chat: bool = True,
    execution_device: str | torch.device = "cpu",
    head_chunk_bytes: int = 8 * 1024 * 1024,
    progress: Optional[ProgressCallback] = None,
    layer_callback=None,
    stream_callback: Optional[Callable[[str], None]] = None,
    token_callback: Optional[Callable[[int, str], None]] = None,
    config=None,
    resident_backbone: bool = True,
    expert_cache: Optional[Any] = None,
    vram_expert_capacity: int = 144,
    ram_expert_capacity: int = 384,
    quantize_ram: bool = False,
    quant_bits: int = 4,
    commit_routing: bool = False,
    commit_threshold: float = 0.15,
) -> DomainSliceGenerateResult:
    """Generate multi-token response for a prompt from on-demand paged OLMoE weights."""
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig

    from pockettitan.metadata.repo import fetch_model_config
    from pockettitan.domainslice.fast_cache import ExpertMemoryCache

    device = torch.device(execution_device)

    # 1. Resolve configuration
    if config is None:
        raw_config = fetch_model_config(
            remote.model_revision.repo_id,
            token=remote.token,
            revision=remote.model_revision.commit_sha,
        )
        config = OlmoeConfig(**raw_config)
    elif isinstance(config, dict):
        config = OlmoeConfig(**config)

    # 2. Resolve tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            remote.model_revision.repo_id,
            revision=remote.model_revision.commit_sha,
            token=remote.token,
            trust_remote_code=True,
        )

    # 3. Format prompt
    formatted_prompt = prompt
    if chat and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = prompt

    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt encoded to empty token sequence")

    eos_token_ids: Set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_id = tokenizer.eos_token_id
        if isinstance(eos_id, (list, tuple, set)):
            eos_token_ids.update(eos_id)
        else:
            eos_token_ids.add(int(eos_id))
    if getattr(config, "eos_token_id", None) is not None:
        eos_id = config.eos_token_id
        if isinstance(eos_id, (list, tuple, set)):
            eos_token_ids.update(eos_id)
        else:
            eos_token_ids.add(int(eos_id))

    # 4. Initialize Runner and Cache
    if expert_cache is None and resident_backbone:
        expert_cache = ExpertMemoryCache(
            vram_capacity=vram_expert_capacity if device.type == "cuda" else 0,
            ram_capacity=ram_expert_capacity,
            quantize_ram=quantize_ram,
            quant_bits=quant_bits,
        )

    runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        remote.model_revision,
        execution_device=device,
        compute_dtype=torch.bfloat16,
        head_chunk_bytes=head_chunk_bytes,
        progress=progress,
        expert_backend="paged",
        resident_backbone=resident_backbone,
        expert_cache=expert_cache,
        commit_routing=commit_routing,
        commit_threshold=commit_threshold,
    )
    kv_cache = DynamicCache(config=config)

    total_start_time = time.perf_counter()
    prefill_start_time = time.perf_counter()

    res = DomainSliceGenerateResult(
        prompt=prompt,
        formatted_prompt=formatted_prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=[],
        generated_text="",
    )

    def accumulate_pass(metrics: SequentialPassMetrics) -> None:
        res.global_page_hits += metrics.global_page_hits
        res.global_page_faults += metrics.global_page_faults
        res.global_remote_bytes += metrics.global_remote_bytes
        res.backbone_page_hits += metrics.backbone_page_hits
        res.backbone_page_faults += metrics.backbone_page_faults
        res.backbone_remote_bytes += metrics.backbone_remote_bytes
        res.expert_page_hits += metrics.expert_page_hits
        res.expert_page_faults += metrics.expert_page_faults
        res.expert_remote_bytes += metrics.expert_remote_bytes
        res.experts_executed += metrics.experts_executed
        res.total_remote_bytes += metrics.total_remote_bytes
        res.total_page_bytes += metrics.logical_page_bytes
        res.peak_projection_bytes = max(res.peak_projection_bytes, metrics.peak_projection_bytes)
        res.peak_head_chunk_bytes = max(res.peak_head_chunk_bytes, metrics.peak_head_chunk_bytes)
        res.peak_cuda_bytes = max(res.peak_cuda_bytes, metrics.peak_cuda_bytes)
        res.peak_rss_bytes = max(res.peak_rss_bytes, metrics.peak_rss_bytes)
        res.final_kv_cache_bytes = max(res.final_kv_cache_bytes, metrics.kv_cache_bytes)

    # 5. Prefill phase
    last_logits = None
    for pos, tok_id in enumerate(prompt_tokens):
        logits, pass_metrics = runner.run(
            tok_id,
            position_id=pos,
            past_key_values=kv_cache,
            use_cache=True,
            layer_callback=(
                (lambda item, p=pos: layer_callback("prefill", p, item))
                if layer_callback is not None
                else None
            ),
        )
        accumulate_pass(pass_metrics)
        last_logits = logits

    res.prefill_elapsed_s = time.perf_counter() - prefill_start_time
    res.prefill_tok_per_s = (
        len(prompt_tokens) / res.prefill_elapsed_s if res.prefill_elapsed_s > 0 else 0.0
    )

    # 6. Autoregressive decode phase
    decode_start_time = time.perf_counter()
    current_logits = last_logits
    generated_ids: List[int] = []

    for step in range(max_new_tokens):
        pos = len(prompt_tokens) + step
        next_tok = _sample_next_token(current_logits, temperature=temperature, top_p=top_p)

        if next_tok in eos_token_ids:
            break

        generated_ids.append(next_tok)
        chunk = tokenizer.decode([next_tok], skip_special_tokens=False)

        if stream_callback is not None:
            stream_callback(chunk)
        if token_callback is not None:
            token_callback(next_tok, chunk)

        if step == max_new_tokens - 1:
            break

        current_logits, pass_metrics = runner.run(
            next_tok,
            position_id=pos,
            past_key_values=kv_cache,
            use_cache=True,
            layer_callback=(
                (lambda item, p=pos: layer_callback("decode", p, item))
                if layer_callback is not None
                else None
            ),
        )
        accumulate_pass(pass_metrics)

    res.decode_elapsed_s = time.perf_counter() - decode_start_time
    res.decode_tok_per_s = (
        len(generated_ids) / res.decode_elapsed_s if res.decode_elapsed_s > 0 else 0.0
    )
    res.total_elapsed_s = time.perf_counter() - total_start_time
    total_tokens = len(prompt_tokens) + len(generated_ids)
    res.overall_tok_per_s = (
        total_tokens / res.total_elapsed_s if res.total_elapsed_s > 0 else 0.0
    )
    res.generated_tokens = generated_ids
    res.generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if expert_cache is not None:
        res.vram_cache_hits = expert_cache.vram_hits
        res.ram_cache_hits = expert_cache.ram_hits

    return res
