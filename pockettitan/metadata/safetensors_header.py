"""Fast binary header parser for Safetensors files (Milestone 0)."""

import json
import re
import struct
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union


class CancellationSignal(Protocol):
    """Small subset shared by ``threading.Event`` and test doubles."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: Optional[float] = None) -> bool: ...


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


class RangeResponseError(ValueError):
    """An HTTP origin did not honor the exact byte range PocketTitan requested."""


_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(?:\d+|\*)$", re.IGNORECASE)


def _validate_range_response(response, requested_start: int, requested_end: int) -> None:
    """Fail closed if a server ignores or rewrites an exact Range request."""
    status = getattr(response, "status", None) or response.getcode()
    if status != 206:
        raise RangeResponseError(
            f"Remote server ignored byte range [{requested_start}, {requested_end}] "
            f"and returned HTTP {status}; refusing a possible full-shard download"
        )
    value = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE_RE.match(value.strip())
    if match is None:
        raise RangeResponseError(f"Malformed or missing Content-Range header: {value!r}")
    actual_start, actual_end = int(match.group(1)), int(match.group(2))
    if (actual_start, actual_end) != (requested_start, requested_end):
        raise RangeResponseError(
            f"Requested bytes [{requested_start}, {requested_end}] but server returned "
            f"Content-Range [{actual_start}, {actual_end}]"
        )


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
        _validate_range_response(resp, 0, probe_size - 1)
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
        _validate_range_response(full_resp, 8, total_header_bytes - 1)
        header_json_bytes = full_resp.read()
        header_json_str = header_json_bytes.decode("utf-8")
        return json.loads(header_json_str), total_header_bytes


class IncompleteRangeRead(IOError):
    """The server returned fewer bytes than the requested range, and retries ran out."""


def stream_remote_bytes(
    url: str,
    byte_start: int,
    byte_end: int,
    write_callback: Callable[[bytes], None],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    chunk_callback: Optional[Callable[[int, int], None]] = None,
    max_attempts: int = 6,
    backoff_seconds: float = 1.5,
    cancel_event: Optional[CancellationSignal] = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> int:
    """Stream an exact, resumable HTTP range into a bounded caller-owned sink.

    The sink is invoked once per chunk and is never retained by this function.
    The response must be ``206`` with an exact ``Content-Range``; accepting a
    ``200`` here could turn a 12 MiB expert fault into a multi-gigabyte shard
    download.
    """
    import time

    total_expected = (byte_end - byte_start) + 1
    if total_expected <= 0:
        return 0
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    opener = urllib.request.build_opener(RedirectRangeHandler)
    received = 0
    last_error: Optional[BaseException] = None

    for attempt in range(max_attempts):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError(f"Cancelled range fetch for {url}")
        if received >= total_expected:
            break
        if attempt:
            delay = backoff_seconds * (2 ** (attempt - 1))
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    raise InterruptedError(f"Cancelled range fetch for {url}")
            else:
                time.sleep(delay)

        request_start = byte_start + received
        req_headers = {
            "User-Agent": "PocketTitan/0.1.0",
            "Range": f"bytes={request_start}-{byte_end}",
        }
        if headers:
            req_headers.update(headers)
        request = urllib.request.Request(url, headers=req_headers)

        try:
            response = opener.open(request, timeout=timeout)
        except OSError as exc:
            last_error = exc
            continue

        with response:
            _validate_range_response(response, request_start, byte_end)
            while received < total_expected:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError(f"Cancelled range fetch for {url}")
                try:
                    block = response.read(min(chunk_size, total_expected - received))
                except OSError as exc:
                    last_error = exc
                    break
                if not block:
                    break
                # Keep sink failures distinct from network failures: replaying a
                # block after a disk error would duplicate data.
                write_callback(block)
                received += len(block)
                if chunk_callback is not None:
                    chunk_callback(len(block), total_expected)

    if received != total_expected:
        raise IncompleteRangeRead(
            f"{url}: got {received:,} of {total_expected:,} bytes for range "
            f"[{byte_start}, {byte_end}] after {max_attempts} attempts"
            + (f"; last error: {last_error!r}" if last_error else "")
        )
    return received


def fetch_remote_bytes(
    url: str,
    byte_start: int,
    byte_end: int,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    chunk_callback: Optional[Callable[[int, int], None]] = None,
    max_attempts: int = 6,
    backoff_seconds: float = 1.5,
    cancel_event: Optional[CancellationSignal] = None,
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
    chunks: list[bytes] = []
    stream_remote_bytes(
        url,
        byte_start,
        byte_end,
        chunks.append,
        headers=headers,
        timeout=timeout,
        chunk_callback=chunk_callback,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        cancel_event=cancel_event,
        chunk_size=4 * 1024 * 1024,
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
