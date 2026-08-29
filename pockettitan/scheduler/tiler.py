"""Memory-bounded Micro-Tiler for executing quantization under hard VRAM constraints."""

from typing import Optional, Tuple
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig
from pockettitan.quantizers.base import BaseQuantizer, QuantizedResult
from pockettitan.scheduler.budget import compute_work_unit_bounds


class MatrixTiler:
    """Decomposes and executes quantization on arbitrarily large matrices under strict VRAM caps."""

    def __init__(self, budget: MemoryBudgetConfig):
        self.budget = budget

    def quantize_matrix(
        self,
        weight: torch.Tensor,
        quantizer: BaseQuantizer,
        hessian: Optional[torch.Tensor] = None,
        source_device: str = "cpu",
        target_device: str = "cuda",
    ) -> Tuple[QuantizedResult, float]:
        """Quantize a 2D weight matrix while strictly enforcing VRAM budget.
        
        Returns:
            (quantized_result, peak_vram_mb_used)
        """
        orig_shape = weight.shape
        orig_dtype = weight.dtype
        out_features, in_features = orig_shape[0], orig_shape[1]
        
        # Determine execution device
        exec_device = target_device if (target_device == "cuda" and torch.cuda.is_available()) else "cpu"
        
        if exec_device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        bounds = compute_work_unit_bounds(
            matrix_shape=[out_features, in_features],
            budget=self.budget,
            quant_config=quantizer.config,
            source_dtype=str(orig_dtype).replace("torch.", ""),
            workspace_multiplier=quantizer.capabilities.workspace_multiplier,
        )
        
        tile_rows = bounds["tile_rows"]
        num_tiles = bounds["num_tiles"]
        
        h_gpu = hessian.to(exec_device) if (hessian is not None and exec_device == "cuda") else hessian
        
        if not bounds["needs_tiling"] or num_tiles <= 1:
            # Fits in single pass
            w_gpu = weight.to(exec_device)
            res = quantizer.quantize(w_gpu, hessian=h_gpu)
            
            peak_vram_mb = 0.0
            if exec_device == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                # Move packed results to CPU host staging to free GPU
                res.packed_weights = res.packed_weights.cpu()
                res.scales = res.scales.cpu()
                if res.zeros is not None:
                    res.zeros = res.zeros.cpu()
                if res.codebook is not None:
                    res.codebook = res.codebook.cpu()
                del w_gpu, h_gpu
                torch.cuda.empty_cache()
            return res, peak_vram_mb

        # Tiled execution: process row chunks sequentially
        packed_tiles = []
        scale_tiles = []
        zero_tiles = []
        peak_vram_mb = 0.0
        
        for i in range(num_tiles):
            r_start = i * tile_rows
            r_end = min(out_features, (i + 1) * tile_rows)
            
            # Slice row chunk and move ONLY this tile to GPU
            tile_cpu = weight[r_start:r_end, :]
            tile_gpu = tile_cpu.to(exec_device)
            
            tile_res = quantizer.quantize(tile_gpu, hessian=h_gpu)
            
            if exec_device == "cuda":
                tile_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_mb = max(peak_vram_mb, tile_peak)
                
            # Move tile results to CPU staging immediately
            packed_tiles.append(tile_res.packed_weights.cpu())
            scale_tiles.append(tile_res.scales.cpu())
            if tile_res.zeros is not None:
                zero_tiles.append(tile_res.zeros.cpu())
                
            # Release GPU tile
            del tile_gpu
            del tile_res
            if exec_device == "cuda":
                torch.cuda.empty_cache()

        # Release Hessian on GPU
        del h_gpu
        if exec_device == "cuda":
            torch.cuda.empty_cache()

        # Stitch tiles together in host RAM
        combined_packed = torch.cat(packed_tiles, dim=0)
        combined_scales = torch.cat(scale_tiles, dim=0)
        combined_zeros = torch.cat(zero_tiles, dim=0) if zero_tiles and zero_tiles[0] is not None else None
        
        res = QuantizedResult(
            packed_weights=combined_packed,
            scales=combined_scales,
            zeros=combined_zeros,
            codebook=None,
            quant_config=quantizer.config,
            original_shape=orig_shape,
            original_dtype=orig_dtype,
            bit_width=float(quantizer.config.bits if quantizer.config.method != "ternary" else 1.58),
            device=exec_device,
        )
        return res, peak_vram_mb
