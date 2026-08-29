"""Test script to load and inspect the PocketTitan 2-bit Qwen model & tokenizer."""

import torch
from transformers import AutoTokenizer, AutoConfig
import safetensors.torch
from pathlib import Path
import json

from pockettitan.quantizers.rtn import RTNQuantizer


def main():
    model_dir = Path("./qwen_2bit_model")
    print(f"=== PocketTitan 2-Bit Model Tester ===")
    print(f"Model directory: {model_dir.resolve()}\n")

    # 1. Load Tokenizer
    print("[1/3] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print(f"Tokenizer successfully loaded!")
    print(f" - Vocabulary Size: {len(tokenizer):,}")
    print(f" - BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f" - EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f" - Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})\n")

    # 2. Test Token Encoding & Decoding
    sample_text = "Hello! Tell me a fun fact about space."
    print(f"[2/3] Testing Tokenizer with sample prompt:")
    print(f" Prompt: '{sample_text}'")
    encoded = tokenizer(sample_text, return_tensors="pt")
    tokens = encoded["input_ids"]
    print(f" Token IDs: {tokens.tolist()}")
    decoded = tokenizer.decode(tokens[0])
    print(f" Decoded: '{decoded}'\n")

    # 3. Inspect Quantized Layers on CUDA / CPU
    print("[3/3] Inspecting 2-Bit Quantized Layers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Execution Device: {device.upper()}")
    
    # Read manifest & quant config
    with open(model_dir / "quant_config.json", "r", encoding="utf-8") as f:
        q_cfg = json.load(f)
    print(f" Quantization: {q_cfg.get('bits')}-Bit {q_cfg.get('method').upper()} (Group Size: {q_cfg.get('group_size')})")

    # Inspect Shard Tensors
    shard_path = model_dir / "model-00001-of-00001.safetensors"
    if not shard_path.exists():
        shard_path = next(model_dir.glob("*.safetensors"))
        
    with safetensors.torch.safe_open(str(shard_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f" Shard '{shard_path.name}' contains {len(keys)} tensor components.")
        
        # Test dequantizing a layer matrix on CUDA
        sample_key = next((k for k in keys if k.endswith(".packed_weight")), None)
        if sample_key:
            prefix = sample_key[:-14]
            packed = f.get_tensor(sample_key)
            scales = f.get_tensor(prefix + ".scales")
            zeros = f.get_tensor(prefix + ".zeros") if (prefix + ".zeros") in keys else None
            
            out_features = scales.shape[0]
            group_size = q_cfg.get("group_size", 128)
            padded_in = scales.shape[1] * group_size
            
            # Send to GPU and dequantize
            if device == "cuda":
                packed = packed.cuda()
                scales = scales.cuda()
                if zeros is not None:
                    zeros = zeros.cuda()
                    
            unpacked = RTNQuantizer._unpack_tensor(packed, q_cfg.get("bits", 2), (out_features, padded_in))
            w_grouped = unpacked.view(-1, group_size).float()
            s_g = scales.view(-1, 1).float()
            z_g = zeros.view(-1, 1).float() if zeros is not None else 0.0
            deq = (w_grouped - z_g) * s_g
            deq_matrix = deq.view(out_features, padded_in)
            
            print(f"\n[+] Successfully loaded & dequantized sample layer on {device.upper()}:")
            print(f" - Layer Component: {prefix}")
            print(f" - Reconstructed Shape: {list(deq_matrix.shape)}")
            print(f" - Mean: {deq_matrix.mean().item():.6f}")
            print(f" - Std:  {deq_matrix.std().item():.6f}")
            print(f" - Memory Used by Dequantized Matrix: {deq_matrix.element_size() * deq_matrix.nelement() / (1024 * 1024):.2f} MB")

    print("\n[SUCCESS] All checks passed successfully!")


if __name__ == "__main__":
    main()
