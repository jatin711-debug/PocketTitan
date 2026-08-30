"""Hardware execution and I/O latency model for out-of-core MoE inference (R2)."""

from typing import Optional
from pydantic import BaseModel


class HardwareProfile(BaseModel):
    """System hardware specifications for out-of-core performance modeling."""

    name: str = "RTX 3050 Laptop + PCIe Gen4 SSD"
    ssd_bandwidth_gbps: float = 3.5  # Sequential/batched read throughput
    ssd_latency_us: float = 25.0  # Base NVMe read latency per batch
    pcie_bandwidth_gbps: float = 7.8  # PCIe 4.0 x4 link bandwidth
    ram_bandwidth_gbps: float = 50.0  # System DDR5 RAM bandwidth
    vram_bandwidth_gbps: float = 192.0  # GPU GDDR6 bandwidth
    gpu_tflops_fp16: float = 9.0  # Peak tensor/FP16 TFLOPS
    gpu_utilization: float = 0.35  # Realistic kernel execution efficiency factor
    
    # Target architecture geometry constants
    expert_size_bytes_4b: int = 2_613_248  # 4-bit stride (~2.49 MiB)
    expert_size_bytes_2b: int = 1_384_448  # 2-bit stride (~1.32 MiB)
    ple_bytes_per_token: int = 65_536  # ~64 KiB
    dense_activated_params: int = 4_312_004_515  # Non-expert activated params


class LatencyBreakdown(BaseModel):
    """Detailed time breakdown for generating one token."""

    token_id: int = 0
    expert_misses: int = 0
    expert_hits: int = 0
    ssd_bytes_read: int = 0
    ssd_time_ms: float = 0.0
    pcie_time_ms: float = 0.0
    gpu_compute_time_ms: float = 0.0
    stall_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0


class HardwareSimulator:
    """Calculates roofline throughput and token latencies under NVMe/RAM/VRAM constraints."""

    def __init__(self, profile: Optional[HardwareProfile] = None):
        self.profile = profile or HardwareProfile()

    def simulate_token(
        self,
        token_id: int,
        expert_misses: int,
        expert_hits: int,
        bits_per_weight: float = 4.0,
        upload_to_gpu: bool = False,
        overlap_io_and_compute: bool = True,
    ) -> LatencyBreakdown:
        """Estimate the latency of a single token decoding step.
        
        Per Design.md §6 and Plan.md §2.2, cold/warm experts execute in RAM on CPU by default.
        VRAM promotion and PCIe upload are only evaluated when upload_to_gpu=True.
        """
        expert_size = (
            self.profile.expert_size_bytes_2b if bits_per_weight <= 2.5 else self.profile.expert_size_bytes_4b
        )
        
        # 1. SSD Read Latency (NVMe Controller -> Host RAM via PCIe DMA)
        ssd_expert_bytes = expert_misses * expert_size
        ssd_total_bytes = ssd_expert_bytes + self.profile.ple_bytes_per_token
        
        ssd_transfer_sec = ssd_total_bytes / (self.profile.ssd_bandwidth_gbps * 1e9)
        ssd_latency_sec = (self.profile.ssd_latency_us * 1e-6) if expert_misses > 0 else 0.0
        ssd_time_ms = (ssd_transfer_sec + ssd_latency_sec) * 1000.0
        
        # 2. Host-to-Device VRAM Upload Latency (if explicitly promoted to GPU VRAM)
        if upload_to_gpu and expert_misses > 0:
            pcie_sec = ssd_expert_bytes / (self.profile.pcie_bandwidth_gbps * 1e9)
            pcie_time_ms = pcie_sec * 1000.0
        else:
            pcie_time_ms = 0.0
            
        # 3. Dense (VRAM/GPU) + Active Expert Compute Latency
        total_active_params = self.profile.dense_activated_params + (expert_misses + expert_hits) * 4_915_200
        flops_per_token = total_active_params * 2.0  # 2 FLOPs per parameter for GEMV
        effective_flops = (self.profile.gpu_tflops_fp16 * 1e12) * self.profile.gpu_utilization
        gpu_compute_ms = (flops_per_token / effective_flops) * 1000.0
        
        # 4. Total Latency Calculation with I/O pipelining
        io_time_ms = ssd_time_ms + pcie_time_ms
        if overlap_io_and_compute:
            stall_time_ms = max(0.0, io_time_ms - gpu_compute_ms)
            total_time_ms = max(io_time_ms, gpu_compute_ms)
        else:
            stall_time_ms = io_time_ms
            total_time_ms = io_time_ms + gpu_compute_ms
            
        tok_s = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
        
        return LatencyBreakdown(
            token_id=token_id,
            expert_misses=expert_misses,
            expert_hits=expert_hits,
            ssd_bytes_read=ssd_total_bytes,
            ssd_time_ms=ssd_time_ms,
            pcie_time_ms=pcie_time_ms,
            gpu_compute_time_ms=gpu_compute_ms,
            stall_time_ms=stall_time_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tok_s,
        )
