"""Base exporter interface for converting quantized checkpoints to inference runtime formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel


class ExportResult(BaseModel):
    format_name: str
    output_path: str
    total_tensors: int
    output_size_bytes: int
    status: str = "success"


class BaseExporter(ABC):
    """Abstract base class for inference runtime exporters."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)

    @abstractmethod
    def export(self, output_path: Union[str, Path]) -> ExportResult:
        """Export the checkpoint to the target format."""
        pass
