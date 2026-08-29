"""Hardware profiling, VRAM budget enforcement, and work unit sizing."""

import ctypes
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod


class CUDADeviceProfile(BaseModel):
    device_id: int
    name: str
    total_vram_mb: float
    free_vram_mb: float
    compute_capability: Tuple[int, int]


class HardwareProfile(BaseModel):
    cuda_available: bool
    devices: List[CUDADeviceProfile] = Field(default_factory=list)
    system_ram_total_mb: float
    system_ram_free_mb: float
    disk_free_mb: float


def get_system_ram_info() -> Tuple[float, float]:
    """Get system RAM info reliably on Windows and Linux without third-party deps."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_mb = stat.ullTotalPhys / (1024 * 1024)
            avail_mb = stat.ullAvailPhys / (1024 * 1024)
            return total_mb, avail_mb
    elif hasattr(os, "sysconf"):
        try:
            pagesize = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total_mb = (total_pages * pagesize) / (1024 * 1024)
            avail_mb = (avail_pages * pagesize) / (1024 * 1024)
            return total_mb, avail_mb
        except Exception:
            pass
    return 16384.0, 8192.0


def get_disk_free_mb(path: Union[str, Path] = ".") -> float:
    """Get free disk space in MiB for target path."""
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 * 1024)
    except Exception:
        return 50000.0


def get_hardware_profile() -> HardwareProfile:
    """Scan and return full system hardware capability and memory profile."""
    cuda_avail = torch.cuda.is_available()
    devices = []
    
    if cuda_avail:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            devices.append(
                CUDADeviceProfile(
                    device_id=i,
                    name=props.name,
                    total_vram_mb=total_b / (1024 * 1024),
                    free_vram_mb=free_b / (1024 * 1024),
                    compute_capability=(props.major, props.minor),
                )
            )
            
    total_ram, free_ram = get_system_ram_info()
    free_disk = get_disk_free_mb(".")
    
    return HardwareProfile(
        cuda_available=cuda_avail,
        devices=devices,
        system_ram_total_mb=total_ram,
        system_ram_free_mb=free_ram,
        disk_free_mb=free_disk,
    )


def apply_cuda_memory_fraction(budget: MemoryBudgetConfig, device_id: int = 0) -> Optional[float]:
    """Set hard CUDA memory fraction to enforce user VRAM ceiling at driver level."""
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        return None
        
    props = torch.cuda.get_device_properties(device_id)
    total_mb = props.total_memory / (1024 * 1024)
    
    fraction = min(1.0, max(0.05, budget.max_vram_mb / total_mb))
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
        return fraction
    except Exception:
        return None


def estimate_tensor_vram_requirement(
    shape: List[int],
    source_dtype: str = "float16",
    quant_method: QuantMethod = QuantMethod.HQQ,
    bits: int = 2,
    workspace_multiplier: float = 2.5,
) -> float:
    """Estimate total VRAM needed in MiB to hold source tensor + workspace intermediates."""
    if not shape:
        return 0.0
    num_elements = 1
    for s in shape:
        num_elements *= s
        
    dtype_bytes = 2
    if "32" in source_dtype:
        dtype_bytes = 4
    elif "8" in source_dtype:
        dtype_bytes = 1
        
    source_bytes = num_elements * dtype_bytes
    working_bytes = source_bytes * workspace_multiplier
    packed_bytes = (num_elements * bits) / 8 + (num_elements / 128) * 4
    
    total_bytes = source_bytes + working_bytes + packed_bytes
    return total_bytes / (1024 * 1024)


def compute_work_unit_bounds(
    matrix_shape: List[int],
    budget: MemoryBudgetConfig,
    quant_config: QuantConfig,
    source_dtype: str = "float16",
    workspace_multiplier: float = 2.5,
) -> Dict[str, Any]:
    """Determine whether a matrix fits in VRAM or compute legal row slicing bounds."""
    full_vram_req_mb = estimate_tensor_vram_requirement(
        shape=matrix_shape,
        source_dtype=source_dtype,
        quant_method=quant_config.method,
        bits=quant_config.bits,
        workspace_multiplier=workspace_multiplier,
    )
    usable_vram_mb = budget.usable_vram_mb
    
    if len(matrix_shape) != 2:
        return {
            "needs_tiling": False,
            "tile_rows": matrix_shape[0] if matrix_shape else 0,
            "num_tiles": 1,
            "estimated_vram_per_tile_mb": round(full_vram_req_mb, 2),
            "total_matrix_vram_mb": round(full_vram_req_mb, 2),
        }
        
    out_features, in_features = matrix_shape[0], matrix_shape[1]
    
    if full_vram_req_mb <= usable_vram_mb:
        return {
            "needs_tiling": False,
            "tile_rows": out_features,
            "num_tiles": 1,
            "estimated_vram_per_tile_mb": round(full_vram_req_mb, 2),
            "total_matrix_vram_mb": round(full_vram_req_mb, 2),
        }
        
    # Calculate exact memory consumption per row
    dtype_bytes = 2
    if "32" in source_dtype:
        dtype_bytes = 4
    elif "8" in source_dtype:
        dtype_bytes = 1
        
    source_bytes_per_row = in_features * dtype_bytes
    working_bytes_per_row = source_bytes_per_row * workspace_multiplier
    packed_bytes_per_row = (in_features * quant_config.bits) / 8 + (in_features / 128) * 4
    total_bytes_per_row = source_bytes_per_row + working_bytes_per_row + packed_bytes_per_row
    vram_per_row_mb = total_bytes_per_row / (1024 * 1024)
    
    # Target 80% of usable VRAM per tile for high stability
    target_tile_vram_mb = usable_vram_mb * 0.80
    raw_tile_rows = max(32, int(target_tile_vram_mb / max(1e-6, vram_per_row_mb)))
    
    # Align tile_rows to 64 for optimal GPU tensor-core throughput
    tile_rows = max(64, (raw_tile_rows // 64) * 64)
    tile_rows = min(out_features, tile_rows)
    
    import math
    num_tiles = math.ceil(out_features / tile_rows)
    est_tile_vram_mb = tile_rows * vram_per_row_mb
    
    return {
        "needs_tiling": True,
        "tile_rows": tile_rows,
        "num_tiles": num_tiles,
        "estimated_vram_per_tile_mb": round(est_tile_vram_mb, 2),
        "total_matrix_vram_mb": round(full_vram_req_mb, 2),
    }
