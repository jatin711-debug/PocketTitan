"""Generate text from a ``.ptitan`` package, and report what it cost to do so.

Kept apart from the CLI so the same entry point is callable from a script or a
notebook, and so the I/O accounting has somewhere to live.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch

from pockettitan.runtime.hf.loader import build_causal_lm, summarize

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class GenerationResult:
    """The text, plus the numbers that say whether it was cheap."""

    prompt: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    seconds: float
    decoded_bytes: int
    decode_calls: int
    resident_bytes: int
    loader: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def decoded_bytes_per_token(self) -> float:
        return self.decoded_bytes / self.generated_tokens if self.generated_tokens else 0.0

    def summary_lines(self) -> List[str]:
        return [
            f"{self.generated_tokens} tokens in {self.seconds:.1f}s "
            f"({self.tokens_per_second:.2f} tok/s)",
            f"{self.decoded_bytes / 2**30:.2f} GiB decoded across "
            f"{self.decode_calls:,} reads "
            f"({self.decoded_bytes_per_token / 2**20:.0f} MiB/token)",
            f"{self.resident_bytes / 2**20:.0f} MiB held in the weight cache",
        ]


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(package_dir: Union[str, Path]):
    """The tokenizer copied into the package at build time."""
    from transformers import AutoTokenizer

    tokenizer_dir = Path(package_dir).resolve() / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"the package has no tokenizer/ directory at {tokenizer_dir}; "
            "rebuild with the tokenizer assets included"
        )
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def generate(
    package_dir: Union[str, Path],
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "auto",
    dtype: str = "float32",
    cache_bytes: int = 512 * 1024 * 1024,
    chat: bool = True,
    on_load: Optional[callable] = None,
) -> GenerationResult:
    """Run one prompt through a package."""
    import time

    resolved_device = resolve_device(device)
    torch_dtype = DTYPES[dtype]

    model, weights = build_causal_lm(
        package_dir,
        device=resolved_device,
        dtype=torch_dtype,
        cache_bytes=cache_bytes,
    )
    try:
        loader_info = summarize(model)
        if on_load is not None:
            on_load(loader_info)

        tokenizer = load_tokenizer(package_dir)
        text = prompt
        if chat and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(resolved_device)

        weights.decoded_bytes = 0
        weights.decode_calls = 0
        started = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started

        produced = out[0, ids.shape[1] :]
        return GenerationResult(
            prompt=prompt,
            text=tokenizer.decode(produced, skip_special_tokens=True),
            prompt_tokens=int(ids.shape[1]),
            generated_tokens=int(produced.numel()),
            seconds=elapsed,
            decoded_bytes=weights.decoded_bytes,
            decode_calls=weights.decode_calls,
            resident_bytes=weights.resident_bytes,
            loader=loader_info,
        )
    finally:
        weights.close()
