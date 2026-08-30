"""Speculative Cross-Layer Lookahead Prefetcher for MoE Expert Paging (Phase R7)."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple
import torch

from pockettitan.runtime.expert.manager import DecodedExpert, ExpertManager


class SpeculativePrefetcher:
    """Predicts Layer L+1 active experts using Layer L hidden states and pre-loads them asynchronously.
    
    Architecture (Plan.md §5 / R7):
    - One small router matvec (~1.3 MB) at Layer L buys a full layer of I/O lead time.
    - Over-fetches m > k (e.g. m=14 vs k=10) to absorb hidden-state representation drift.
    - Strict invariant: await_partial must never make a token slower than no prefetch.
    """

    def __init__(
        self,
        expert_manager: ExpertManager,
        num_prefetch_experts: int = 14,
        max_workers: int = 4,
    ):
        self.manager = expert_manager
        self.m = num_prefetch_experts
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pt-prefetch")
        
        # Pending asynchronous read futures: (layer, expert) -> Future[DecodedExpert]
        self.pending_futures: Dict[Tuple[int, int], Future[DecodedExpert]] = {}

        # Accuracy and drift tracking metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.drift_events = 0

    @property
    def prediction_accuracy(self) -> float:
        return (
            (self.correct_predictions / self.total_predictions)
            if self.total_predictions > 0
            else 0.0
        )

    def predict_next_layer_experts(
        self,
        current_layer: int,
        hidden_state: torch.Tensor,
        next_router_weight: torch.Tensor,
    ) -> List[int]:
        """Compute logits for Layer L+1 router using Layer L post-attention state and select top-m."""
        # hidden_state: [..., hidden_dim], next_router_weight: [num_experts, hidden_dim]
        h = hidden_state.detach().float()
        w = next_router_weight.detach().float()
        
        logits = torch.matmul(h, w.t())
        if logits.ndim > 1:
            logits = logits[-1]  # Take last token step if sequence
            
        top_m = torch.topk(logits, k=min(self.m, logits.shape[-1])).indices.tolist()
        return top_m

    def issue_speculative_prefetch(
        self,
        next_layer: int,
        predicted_experts: Sequence[int],
    ) -> None:
        """Submit background NVMe I/O reads for predicted experts not already cached."""
        for exp_idx in predicted_experts:
            key = (next_layer, exp_idx)
            # If already resident in VRAM or RAM SLRU cache, skip I/O
            if key in self.manager.vram_hot_tier or self.manager.ram_cache.contains(key):
                continue
            if key in self.pending_futures and not self.pending_futures[key].done():
                continue

            # Submit asynchronous read task
            def _async_load(l: int, e: int) -> DecodedExpert:
                raw_bytes = self.manager.read_expert_record(l, e)
                decoded = self.manager.decode_expert_payload(raw_bytes)
                self.manager.ram_cache.put((l, e), decoded)
                return decoded

            self.pending_futures[key] = self.executor.submit(_async_load, next_layer, exp_idx)

    def await_partial(
        self,
        layer: int,
        actual_top_k: Sequence[int],
        timeout_ms: float = 0.0,
    ) -> None:
        """Synchronize available prefetch futures for the actual top-k experts without blocking.
        
        A prefetch must never stall execution if compute finishes ahead of I/O.
        """
        for exp_idx in actual_top_k:
            key = (layer, exp_idx)
            self.total_predictions += 1

            if key in self.pending_futures:
                fut = self.pending_futures.pop(key)
                if fut.done() and not fut.cancelled() and fut.exception() is None:
                    self.correct_predictions += 1
                elif timeout_ms > 0:
                    try:
                        fut.result(timeout=timeout_ms / 1000.0)
                        self.correct_predictions += 1
                    except Exception:
                        self.drift_events += 1
                else:
                    self.drift_events += 1
            else:
                self.drift_events += 1

    def close(self) -> None:
        self.executor.shutdown(wait=False)

    def __enter__(self) -> "SpeculativePrefetcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
