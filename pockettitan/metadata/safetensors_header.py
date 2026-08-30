"""Fast binary header parser for Safetensors files (Milestone 0)."""

import json
import struct
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union


class RedirectRangeHandler(urllib.request.HTTPRedirectHandler):
    """Custom HTTP redirect handler that preserves Range and Authorization headers across 302/307 redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None

        # Copy critical streaming headers across redirect hops
        for header_name in ["Range", "Authorization", "User-Agent"]:
            val = req.get_header(header_name, None)
            if val is not None:
                new_req.add_unredirected_header(header_name, val)
        return new_req


def parse_safetensors_header_from_bytes(data: bytes) -> Tuple[Dict[str, Any], int]:
    """Parse header from in-memory bytes."""
    if len(data) < 8:
        raise ValueError("Data smaller than 8 bytes, invalid Safetensors.")
    header_length = struct.unpack("<Q", data[:8])[0]
    total_header_bytes = 8 + header_length
    if len(data) < total_header_bytes:
        raise ValueError(f"Incomplete header. Expected {header_length} bytes.")
    header_json_str = data[8:total_header_bytes].decode("utf-8")
    return json.loads(header_json_str), total_header_bytes


def parse_local_safetensors_header(file_path: Union[str, Path]) -> Tuple[Dict[str, Any], int]:
    """Read only the first 8 bytes + N header bytes of a local Safetensors file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"File {path} is smaller than 8 bytes, invalid Safetensors.")

        header_length = struct.unpack("<Q", header_len_bytes)[0]
        header_json_bytes = f.read(header_length)
        if len(header_json_bytes) < header_length:
            raise ValueError(f"Incomplete header in {path}. Expected {header_length} bytes.")

        header_json_str = header_json_bytes.decode("utf-8")
        return json.loads(header_json_str), 8 + header_length


def parse_remote_safetensors_header(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    probe_size: int = 131072,  # 128 KiB
) -> Tuple[Dict[str, Any], int]:
    """Read only the header portion of a remote Safetensors file using HTTP Range requests."""
    opener = urllib.request.build_opener(RedirectRangeHandler)
    req_headers = {"User-Agent": "PocketTitan/0.1.0", "Range": f"bytes=0-{probe_size - 1}"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    with opener.open(req, timeout=15) as resp:
        chunk = resp.read()

    if len(chunk) < 8:
        raise ValueError(f"Remote Safetensors file {url} returned less than 8 bytes")

    header_length = struct.unpack("<Q", chunk[:8])[0]
    total_header_bytes = 8 + header_length

    if len(chunk) >= total_header_bytes:
        header_json_str = chunk[8:total_header_bytes].decode("utf-8")
        return json.loads(header_json_str), total_header_bytes

    # If header exceeded 128KB, fetch full header range
    full_headers = dict(req_headers)
    full_headers["Range"] = f"bytes=8-{total_header_bytes - 1}"
    full_req = urllib.request.Request(url, headers=full_headers)
    with opener.open(full_req, timeout=15) as full_resp:
        header_json_bytes = full_resp.read()
        header_json_str = header_json_bytes.decode("utf-8")
        return json.loads(header_json_str), total_header_bytes


class IncompleteRangeRead(IOError):
    """The server returned fewer bytes than the requested range, and retries ran out."""


def fetch_remote_bytes(
    url: str,
    byte_start: int,
    byte_end: int,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    chunk_callback: Optional[Callable[[int, int], None]] = None,
    max_attempts: int = 6,
    backoff_seconds: float = 1.5,
) -> bytes:
    """Fetch an exact byte range, resuming rather than truncating.

    A long transfer over a slow link will drop: the socket closes early, the
    read loop ends without error, and the caller gets a short buffer. Verifying
    only the reshape downstream turns that into
    ``shape '[17408, 5120]' is invalid for input of size 47516081`` three hours
    into a build -- an error that names neither the cause nor the fix.

    So the range is tracked to the byte and a short read re-requests **only the
    missing suffix**, which costs one more request rather than restarting a
    178 MB tensor. Only a genuinely exhausted retry budget raises, and it raises
    saying what went wrong.
    """
    import time

    opener = urllib.request.build_opener(RedirectRangeHandler)
    total_expected = (byte_end - byte_start) + 1
    if total_expected <= 0:
        return b""

    chunks: list = []
    received = 0
    last_error: Optional[BaseException] = None

    for attempt in range(max_attempts):
        if received >= total_expected:
            break
        if attempt:
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))

        req_headers = {
            "User-Agent": "PocketTitan/0.1.0",
            "Range": f"bytes={byte_start + received}-{byte_end}",
        }
        if headers:
            req_headers.update(headers)
        request = urllib.request.Request(url, headers=req_headers)

        try:
            with opener.open(request, timeout=timeout) as response:
                chunk_size = 4 * 1024 * 1024  # 4 MiB
                while received < total_expected:
                    block = response.read(min(chunk_size, total_expected - received))
                    if not block:
                        break
                    chunks.append(block)
                    received += len(block)
                    if chunk_callback is not None:
                        chunk_callback(len(block), total_expected)
        except OSError as exc:  # socket resets, timeouts, truncated TLS records
            last_error = exc

    if received != total_expected:
        raise IncompleteRangeRead(
            f"{url}: got {received:,} of {total_expected:,} bytes for range "
            f"[{byte_start}, {byte_end}] after {max_attempts} attempts"
            + (f"; last error: {last_error!r}" if last_error else "")
        )
    return b"".join(chunks)


def parse_safetensors_header(
    source: Union[str, Path],
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], int]:
    """Universal header parser supporting local paths or HTTP(S) URLs."""
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        return parse_remote_safetensors_header(source_str, headers=headers)
    return parse_local_safetensors_header(source)
