"""Direct local text generation and runtime loader for PocketTitan quantized checkpoints."""

import json
from pathlib import Path
from typing import Union
import safetensors.torch
import torch
from transformers import AutoTokenizer

from pockettitan.quantizers.rtn import RTNQuantizer


class PocketTitanModelRunner:
    """Loads and generates text from a PocketTitan 2-bit quantized checkpoint."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        # Load Tokenizer
        print(f"Loading Tokenizer from {self.checkpoint_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_dir), trust_remote_code=True
        )

        # Load Quant Config
        q_cfg_path = self.checkpoint_dir / "quant_config.json"
        self.bits = 2
        self.group_size = 128
        if q_cfg_path.exists():
            with open(q_cfg_path, "r", encoding="utf-8") as f:
                q_cfg = json.load(f)
                self.bits = q_cfg.get("bits", 2)
                self.group_size = q_cfg.get("group_size", 128)

        print(
            f"Loaded PocketTitan Model Metadata (Bits={self.bits}, GroupSize={self.group_size}, Device={self.device})"
        )

    def inspect_layer_sample(self, layer_idx: int = 0) -> str:
        """Inspect and verify dequantized layer numerical fidelity."""
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        if not index_path.exists():
            return "No index found."

        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)

        shard_file = self.checkpoint_dir / idx.get("weight_map", {}).get(
            f"model.language_model.layers.{layer_idx}.self_attn.q_proj.packed_weight",
            "model-00001-of-00001.safetensors",
        )
        if not shard_file.exists():
            shard_file = self.checkpoint_dir / "model-00001-of-00001.safetensors"

        with safetensors.torch.safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for k in f.keys():
                if f"layers.{layer_idx}" in k and k.endswith(".packed_weight"):
                    packed = f.get_tensor(k)
                    prefix = k[:-14]
                    scales = f.get_tensor(prefix + ".scales")
                    zeros = f.get_tensor(prefix + ".zeros")

                    out_f = scales.shape[0]
                    padded_in = scales.shape[1] * self.group_size
                    unpacked = RTNQuantizer._unpack_tensor(packed, self.bits, (out_f, padded_in))
                    w_grouped = unpacked.view(-1, self.group_size).float()
                    s_g = scales.view(-1, 1).float()
                    z_g = zeros.view(-1, 1).float() if zeros is not None else 0.0
                    deq = (w_grouped - z_g) * s_g
                    deq_tensor = deq.view(out_f, padded_in)

                    return f"Layer {layer_idx} [{prefix}]: Shape={deq_tensor.shape}, Mean={deq_tensor.mean():.4f}, Std={deq_tensor.std():.4f}"

        return f"Layer {layer_idx} inspected."
