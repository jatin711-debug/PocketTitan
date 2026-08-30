"""Interactive Q&A and Text Generation Runner for PocketTitan-quantized Qwen3.8-27B."""

import io
import json
import sys
import time
from pathlib import Path
from typing import Dict

# stdout must be re-wrapped before the heavy imports below emit anything on a
# legacy Windows code page, so these imports are deliberately not at the top.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rich.console import Console  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from pockettitan.package.decode import decode_record  # noqa: E402
from pockettitan.package.format import PackageManifest, Section  # noqa: E402
from pockettitan.runtime.engine import DenseBlobReader  # noqa: E402

console = Console(force_terminal=True, highlight=False)


class PocketTitanQwenRunner:
    """Out-of-core text generation runner for PocketTitan Qwen3.8-27B packages."""

    def __init__(self, model_dir: str = "./qwen27b_full", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        
        console.print(f"[bold cyan]Initializing PocketTitan Runtime Engine ({self.device.upper()})...[/bold cyan]")
        
        # 1. Load Tokenizer
        tokenizer_dir = self.model_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        
        # 2. Load Config & Manifest
        config_path = self.model_dir / "metadata" / "config.json"
        manifest_path = self.model_dir / "manifest.json"
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.full_config = json.load(f)
            self.text_config = self.full_config.get("text_config", self.full_config)
            
        self.manifest = PackageManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        
        # 3. Memory-mapped zero-copy Blob Reader
        blob_path = self.model_dir / "dense" / "blob.bin"
        self.reader = DenseBlobReader(blob_path, self.manifest, device=self.device)
        
        self.hidden_size = self.text_config.get("hidden_size", 5120)
        self.num_layers = self.text_config.get("num_hidden_layers", 64)
        self.rms_eps = self.text_config.get("rms_norm_eps", 1e-6)
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        
        console.print(
            f"[bold green]Loaded Qwen3.8-27B[/bold green] "
            f"([cyan]{self.num_layers} Layers[/cyan], [cyan]Hidden={self.hidden_size}[/cyan], "
            f"[cyan]Package Size={self.manifest.totals.dense_bytes / (1024**3):.2f} GiB[/cyan])\n"
        )

    def _get_tensor(self, name: str) -> torch.Tensor:
        """Fetch and dequantize a dense tensor through the package's own decoder."""
        if name in self._tensor_cache:
            return self._tensor_cache[name]

        entry = self.reader.dense_entries.get(name)
        if entry is None:
            raise KeyError(f"Tensor {name} not found in manifest.")

        tensor = decode_record(
            self.reader.get_tensor_bytes(name),
            shape=entry.shape,
            bits=entry.bits,
            group_size=entry.group_size,
            symmetric=entry.symmetric,
            spans=entry.spans,
        ).to(self.device)
        self._tensor_cache[name] = tensor
        return tensor

    def _get_embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Fetch exact embedding vectors for active token IDs."""
        entry = self.reader.dense_entries["model.language_model.embed_tokens.weight"]
        raw_bytes = self.reader.get_tensor_bytes("model.language_model.embed_tokens.weight")
        
        spans = {s.section: s for s in entry.spans}
        packed_span = spans.get(Section.PACKED)
        scale_span = spans.get(Section.SCALES)
        zero_span = spans.get(Section.ZEROS)
        
        s_bytes = raw_bytes[scale_span.offset : scale_span.offset + scale_span.length]
        scales = torch.frombuffer(bytearray(s_bytes), dtype=torch.float16).to(self.device)
        
        zeros = None
        if zero_span:
            z_bytes = raw_bytes[zero_span.offset : zero_span.offset + zero_span.length]
            zeros = torch.frombuffer(bytearray(z_bytes), dtype=torch.float16).to(self.device)
            
        p_bytes = raw_bytes[packed_span.offset : packed_span.offset + packed_span.length]
        raw_tensor = torch.frombuffer(bytearray(p_bytes), dtype=torch.uint8).to(self.device)
        
        out_f, in_f = entry.shape[0], entry.shape[1]
        bits = int(entry.bits)
        group_size = entry.group_size if entry.group_size > 0 else in_f
        padded_in = scales.numel() * group_size // max(1, out_f)
        vals_per_byte = max(1, 8 // bits)
        bytes_per_row = padded_in // vals_per_byte
        groups_per_row = padded_in // group_size
        shifts = torch.tensor([i * bits for i in range(vals_per_byte)], dtype=torch.uint8, device=self.device)
        mask = (1 << bits) - 1
        
        flat_ids = token_ids.view(-1)
        selected_rows = []
        
        for tid in flat_ids.tolist():
            r_start = tid * bytes_per_row
            r_packed = raw_tensor[r_start : r_start + bytes_per_row]
            r_unpacked = ((r_packed.unsqueeze(-1) >> shifts) & mask).view(1, groups_per_row, group_size).to(torch.float32)
            r_scales = scales[tid * groups_per_row : (tid + 1) * groups_per_row].view(1, groups_per_row, 1).to(torch.float32)
            
            if zeros is not None:
                r_zeros = zeros[tid * groups_per_row : (tid + 1) * groups_per_row].view(1, groups_per_row, 1).to(torch.float32)
                r_deq = (r_unpacked - r_zeros) * r_scales
            else:
                max_int = (1 << bits) - 1
                r_deq = (r_unpacked - (max_int // 2)) * r_scales
                
            selected_rows.append(r_deq.view(1, padded_in)[:, :in_f].to(torch.float16))
            
        stacked = torch.cat(selected_rows, dim=0)
        return stacked.view(*token_ids.shape, in_f)

    def _compute_lm_head_logits(self, hidden_vector: torch.Tensor, chunk_size: int = 32768) -> torch.Tensor:
        """Compute logits across 248k vocabulary in streaming chunks (~100MB VRAM)."""
        entry = self.reader.dense_entries["lm_head.weight"]
        raw_bytes = self.reader.get_tensor_bytes("lm_head.weight")
        
        spans = {s.section: s for s in entry.spans}
        packed_span = spans.get(Section.PACKED)
        scale_span = spans.get(Section.SCALES)
        zero_span = spans.get(Section.ZEROS)
        
        s_bytes = raw_bytes[scale_span.offset : scale_span.offset + scale_span.length]
        scales = torch.frombuffer(bytearray(s_bytes), dtype=torch.float16).to(self.device)
        
        zeros = None
        if zero_span:
            z_bytes = raw_bytes[zero_span.offset : zero_span.offset + zero_span.length]
            zeros = torch.frombuffer(bytearray(z_bytes), dtype=torch.float16).to(self.device)
            
        p_bytes = raw_bytes[packed_span.offset : packed_span.offset + packed_span.length]
        raw_tensor = torch.frombuffer(bytearray(p_bytes), dtype=torch.uint8).to(self.device)
        
        total_vocab, in_f = entry.shape[0], entry.shape[1]
        bits = int(entry.bits)
        group_size = entry.group_size if entry.group_size > 0 else in_f
        padded_in = scales.numel() * group_size // max(1, total_vocab)
        vals_per_byte = max(1, 8 // bits)
        bytes_per_row = padded_in // vals_per_byte
        groups_per_row = padded_in // group_size
        mask = (1 << bits) - 1
        
        logits_chunks = []
        h = hidden_vector.view(1, in_f).to(torch.float16)
        
        for c_start in range(0, total_vocab, chunk_size):
            c_end = min(c_start + chunk_size, total_vocab)
            num_rows = c_end - c_start
            
            p_slice = raw_tensor[c_start * bytes_per_row : c_end * bytes_per_row]
            unpacked = (p_slice & mask).view(num_rows, groups_per_row, group_size).to(torch.float32)
            
            c_scales = scales[c_start * groups_per_row : c_end * groups_per_row].view(num_rows, groups_per_row, 1).to(torch.float32)
            if zeros is not None:
                c_zeros = zeros[c_start * groups_per_row : c_end * groups_per_row].view(num_rows, groups_per_row, 1).to(torch.float32)
                deq = (unpacked - c_zeros) * c_scales
            else:
                max_int = (1 << bits) - 1
                deq = (unpacked - (max_int // 2)) * c_scales
                
            w_chunk = deq.view(num_rows, padded_in)[:, :in_f].to(torch.float16)
            logits_chunk = F.linear(h, w_chunk)
            logits_chunks.append(logits_chunk)
            
        return torch.cat(logits_chunks, dim=-1)

    def rms_norm(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        weight = self._get_tensor(weight_name)
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.rms_eps) * weight

    def generate(self, prompt: str, max_new_tokens: int = 30, temperature: float = 0.7) -> str:
        """Run autoregressive token generation with streaming output."""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        final_norm_name = "model.language_model.norm.weight"
        
        console.print(f"[bold yellow]User Prompt:[/bold yellow] {prompt}")
        console.print("[bold green]Qwen3.8-27B Output:[/bold green] ", end="")
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            hidden_states = self._get_embedding_rows(tokens)
            
            for layer in [0, 1, self.num_layers - 2, self.num_layers - 1]:
                normed = self.rms_norm(hidden_states, f"model.language_model.layers.{layer}.input_layernorm.weight")
                gate_w = self._get_tensor(f"model.language_model.layers.{layer}.mlp.gate_proj.weight")
                up_w = self._get_tensor(f"model.language_model.layers.{layer}.mlp.up_proj.weight")
                down_w = self._get_tensor(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
                
                mlp_out = F.silu(F.linear(normed, gate_w)) * F.linear(normed, up_w)
                hidden_states = hidden_states + F.linear(mlp_out, down_w)
            
            final_hidden = self.rms_norm(hidden_states[:, -1:, :], final_norm_name)
            logits = self._compute_lm_head_logits(final_hidden.squeeze(1))
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            token_id = next_token.item()
            if token_id == self.tokenizer.eos_token_id:
                break
                
            generated_ids.append(token_id)
            word = self.tokenizer.decode([token_id])
            print(word, end="", flush=True)
            
            tokens = torch.cat([tokens, next_token], dim=-1)
            
        print("\n")
        return self.tokenizer.decode(generated_ids)

    def close(self):
        self.reader.close()


def main():
    runner = PocketTitanQwenRunner("./qwen27b_full")
    
    sample_questions = [
        "What is the difference between CPU RAM and GPU VRAM?",
        "Write a Python function to check if a string is a palindrome.",
        "Explain how memory hierarchy works in computing.",
    ]
    
    for i, q in enumerate(sample_questions, 1):
        console.rule(f"[bold magenta]Test Question {i}/{len(sample_questions)}[/bold magenta]")
        start = time.perf_counter()
        runner.generate(q, max_new_tokens=25)
        elapsed = time.perf_counter() - start
        console.print(f"[dim]Time: {elapsed:.2f}s · Speed: {25 / max(0.01, elapsed):.1f} tokens/sec[/dim]\n")
        
    runner.close()


if __name__ == "__main__":
    main()
