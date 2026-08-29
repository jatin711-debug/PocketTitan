"""Memory-bounded Micro-Tiler executing post-training quantization under hard VRAM budgets."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, TensorAddress
from pockettitan.models.layout import Dense2DLayout, FusedExperts3DLayout, get_layout_adapter
from pockettitan.quantizers.base import BaseQuantizer, QuantizedResult
from pockettitan.scheduler.budget import compute_work_unit_bounds


class MatrixTiler:
    """Decomposes and executes quantization on arbitrarily large matrices under strict VRAM caps."""

    def __init__(self, budget: MemoryBudgetConfig):
        self.budget = budget

    def quantize_address(
        self,
        reader: Any,
        tensor_addr: TensorAddress,
        quantizer: BaseQuantizer,
        hessian: Optional[torch.Tensor] = None,
        target_device: str = "cuda",
        chunk_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[QuantizedResult, float]:
        """Stream and quantize a tensor directly from its address without materializing oversized tensors in RAM.
        
        Calculates work unit row bounds BEFORE fetching tensor bytes over memory-map or network.
        """
        shape = tensor_addr.shape
        name = tensor_addr.name
        dtype_str = tensor_addr.dtype
        layout = get_layout_adapter(name, shape, dtype_str)
        
        exec_device = target_device if (target_device == "cuda" and torch.cuda.is_available()) else "cpu"
        
        # 1. Handle Fused 3-D Experts Layout ([num_experts, out_features, in_features])
        if isinstance(layout, FusedExperts3DLayout):
            num_experts = layout.get_num_subunits()
            expert_shape = layout.get_subunit_shape(0)
            
            packed_experts = []
            scale_experts = []
            zero_experts = []
            peak_vram_mb = 0.0
            
            for exp_idx in range(num_experts):
                # Read single expert 2-D slice directly from reader without loading 3-D bank
                if hasattr(reader, "read_3d_expert_slice"):
                    exp_tensor = reader.read_3d_expert_slice(tensor_addr, exp_idx)
                else:
                    full_3d = reader.read_tensor(tensor_addr)
                    exp_tensor = layout.extract_subunit_tensor(full_3d, exp_idx)
                    del full_3d
                    
                exp_res, exp_peak = self.quantize_matrix(
                    exp_tensor,
                    quantizer=quantizer,
                    hessian=hessian,
                    target_device=exec_device,
                )
                peak_vram_mb = max(peak_vram_mb, exp_peak)
                
                packed_experts.append(exp_res.packed_weights)
                scale_experts.append(exp_res.scales)
                if exp_res.zeros is not None:
                    zero_experts.append(exp_res.zeros)
                del exp_tensor, exp_res
                
            combined_packed = torch.stack(packed_experts, dim=0)
            combined_scales = torch.stack(scale_experts, dim=0)
            combined_zeros = torch.stack(zero_experts, dim=0) if zero_experts and zero_experts[0] is not None else None
            
            res = QuantizedResult(
                packed_weights=combined_packed,
                scales=combined_scales,
                zeros=combined_zeros,
                codebook=None,
                quant_config=quantizer.config,
                original_shape=tuple(shape),
                original_dtype=torch.float16,
                bit_width=float(quantizer.config.bits if quantizer.config.method != "ternary" else 1.58),
                device=exec_device,
            )
            return res, peak_vram_mb

        # 2. Standard 2-D Matrix Layout
        out_features, in_features = shape[0], shape[1] if len(shape) > 1 else 1
        bounds = compute_work_unit_bounds(
            matrix_shape=[out_features, in_features],
            budget=self.budget,
            quant_config=quantizer.config,
            source_dtype=dtype_str,
            workspace_multiplier=quantizer.capabilities.workspace_multiplier,
        )
        
        # If matrix fits within budget without slicing, fetch whole tensor
        if not bounds["needs_tiling"] or bounds["num_tiles"] <= 1:
            tensor_data = reader.read_tensor(tensor_addr, chunk_callback=chunk_callback)
            return self.quantize_matrix(
                tensor_data,
                quantizer=quantizer,
                hessian=hessian,
                target_device=exec_device,
            )

        # 3. Memory-Bounded Sliced Streaming: Fetch only row chunks sequentially
        tile_rows = bounds["tile_rows"]
        num_tiles = bounds["num_tiles"]
        
        packed_tiles = []
        scale_tiles = []
        zero_tiles = []
        peak_vram_mb = 0.0
        
        for i in range(num_tiles):
            r_start = i * tile_rows
            r_end = min(out_features, (i + 1) * tile_rows)
            
            # Read sub-slice directly via zero-copy slice or HTTP range
            if hasattr(reader, "read_slice"):
                tile_tensor = reader.read_slice(tensor_addr, r_start, r_end)
            else:
                full_t = reader.read_tensor(tensor_addr)
                tile_tensor = full_t[r_start:r_end, :]
                del full_t
                
            tile_res, tile_peak = self.quantize_matrix(
                tile_tensor,
                quantizer=quantizer,
                hessian=hessian,
                target_device=exec_device,
            )
            peak_vram_mb = max(peak_vram_mb, tile_peak)
            
            packed_tiles.append(tile_res.packed_weights)
            scale_tiles.append(tile_res.scales)
            if tile_res.zeros is not None:
                zero_tiles.append(tile_res.zeros)
            del tile_tensor, tile_res
            
        combined_packed = torch.cat(packed_tiles, dim=0)
        combined_scales = torch.cat(scale_tiles, dim=0)
        combined_zeros = torch.cat(zero_tiles, dim=0) if zero_tiles and zero_tiles[0] is not None else None
        
        res = QuantizedResult(
            packed_weights=combined_packed,
            scales=combined_scales,
            zeros=combined_zeros,
            codebook=None,
            quant_config=quantizer.config,
            original_shape=tuple(shape),
            original_dtype=torch.float16,
            bit_width=float(quantizer.config.bits if quantizer.config.method != "ternary" else 1.58),
            device=exec_device,
        )
        return res, peak_vram_mb

    def quantize_matrix(
        self,
        weight: torch.Tensor,
        quantizer: BaseQuantizer,
        hessian: Optional[torch.Tensor] = None,
        source_device: str = "cpu",
        target_device: str = "cuda",
    ) -> Tuple[QuantizedResult, float]:
        """Quantize an already-loaded 2D weight matrix while strictly enforcing VRAM budget."""
        orig_shape = weight.shape
        orig_dtype = weight.dtype
        out_features, in_features = orig_shape[0], orig_shape[1] if len(orig_shape) > 1 else 1
        
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
            w_gpu = weight.to(exec_device)
            res = quantizer.quantize(w_gpu, hessian=h_gpu)
            
            peak_vram_mb = 0.0
            if exec_device == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
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
            
            tile_cpu = weight[r_start:r_end, :]
            tile_gpu = tile_cpu.to(exec_device)
            
            tile_res = quantizer.quantize(tile_gpu, hessian=h_gpu)
            
            if exec_device == "cuda":
                tile_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_mb = max(peak_vram_mb, tile_peak)
                
            packed_tiles.append(tile_res.packed_weights.cpu())
            scale_tiles.append(tile_res.scales.cpu())
            if tile_res.zeros is not None:
                zero_tiles.append(tile_res.zeros.cpu())
                
            del tile_gpu, tile_res
            if exec_device == "cuda":
                torch.cuda.empty_cache()

        del h_gpu
        if exec_device == "cuda":
            torch.cuda.empty_cache()

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
