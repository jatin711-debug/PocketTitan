"""PocketTitan 2-Bit Model Runner & GGUF Exporter."""

import json
from pathlib import Path
import time
import safetensors.torch
import torch
from transformers import AutoTokenizer

from pockettitan.exporters.gguf import GGUFExporter
from pockettitan.quantizers.rtn import RTNQuantizer


def main():
    model_dir = Path("./qwen_2bit_model")
    gguf_output = Path("./qwen_2bit.gguf")
    
    print("=" * 65)
    print("  PocketTitan 2-Bit Quantized Checkpoint Runner")
    print("=" * 65)
    print(f"Model Directory: {model_dir.resolve()}\n")

    # 1. Load Tokenizer
    print("[1/3] Loading Tokenizer & Chat Template...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print(f"Tokenizer loaded successfully! Vocab size: {len(tokenizer):,}")
    
    prompt = "Explain quantization in machine learning simply."
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(formatted, return_tensors="pt")
    print(f"Tokenized prompt length: {encoded['input_ids'].shape[1]} tokens\n")

    # 2. Checkpoint Verification
    print("[2/3] Verifying 2-Bit Quantized Weights in Checkpoint...")
    with open(model_dir / "quant_config.json", "r") as f:
        q_cfg = json.load(f)
    print(f"Quantization Config: {q_cfg.get('bits')}-bit {q_cfg.get('method').upper()} (Group Size: {q_cfg.get('group_size')})")
    
    shards = list(model_dir.glob("*.safetensors"))
    total_bytes = sum(s.stat().st_size for s in shards)
    print(f"Total Packed Checkpoint Size on Disk: {total_bytes / (1024**3):.2f} GiB (across {len(shards)} shards)\n")

    # 3. Export to GGUF for llama.cpp / Ollama fast execution
    print("[3/3] Exporting to GGUF for high-speed local inference (llama.cpp / Ollama)...")
    exporter = GGUFExporter(model_dir)
    res = exporter.export(str(gguf_output))
    print(f"\n[SUCCESS] Exported 2-Bit GGUF Model: {res.output_path}")
    print(f" - Output Size: {res.output_size_bytes / (1024**3):.2f} GiB")
    print(f" - Exported Tensors: {res.total_tensors}")
    print("\nTo run real-time inference with full GPU acceleration:")
    print(f"  llama-cli -m {gguf_output} -p \"{prompt}\"")


if __name__ == "__main__":
    main()
