"""Integrity primitives used by the package writer and validator."""

import hashlib
import os
from pathlib import Path


_CRC32C_POLYNOMIAL = 0x82F63B78


def _crc32c_table() -> tuple[int, ...]:
    values = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLYNOMIAL if crc & 1 else 0)
        values.append(crc)
    return tuple(values)


_CRC32C_TABLE = _crc32c_table()

try:
    import google_crc32c

    def crc32c(data: bytes, crc: int = 0) -> int:
        if crc == 0:
            return google_crc32c.value(data)
        checksum = google_crc32c.Checksum(crc.to_bytes(4, "big"))
        checksum.update(data)
        return checksum.value

except ImportError:
    try:
        import crc32c as _c_crc32c

        def crc32c(data: bytes, crc: int = 0) -> int:
            return _c_crc32c.crc32c(data, crc)

    except ImportError:
        def crc32c(data: bytes, crc: int = 0) -> int:
            """Return the Castagnoli CRC-32C of ``data`` (pure-Python fallback)."""
            value = crc ^ 0xFFFFFFFF
            for byte in data:
                value = _CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
            return value ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return f"{crc32c(data):08x}"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def durable_replace(path: Path, payload: bytes) -> None:
    """Atomically replace ``path`` after flushing the replacement to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
