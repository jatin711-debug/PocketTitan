"""Fast Safetensors header parser for local files and remote HTTP ranges."""

import json
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class RedirectRangeHandler(urllib.request.HTTPRedirectHandler):
    """Preserve HTTP Range headers when following CDN redirects."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return urllib.request.Request(
            newurl,
            headers={
                "Range": req.get_header("Range"),
                "User-Agent": "PocketTitan/0.1.0",
            },
        )


def parse_safetensors_header_from_bytes(raw_bytes: bytes) -> Tuple[Dict[str, Any], int]:
    """Parse Safetensors JSON header from raw byte sequence.
    
    Returns:
        (header_dict, total_header_bytes) where total_header_bytes = 8 + header_length.
    """
    if len(raw_bytes) < 8:
        raise ValueError(f"Payload too short for Safetensors header: {len(raw_bytes)} bytes")
    
    header_length = struct.unpack("<Q", raw_bytes[:8])[0]
    total_header_bytes = 8 + header_length
    
    if len(raw_bytes) < total_header_bytes:
        raise ValueError(
            f"Buffer contains {len(raw_bytes)} bytes, but header requires {total_header_bytes} bytes"
        )
    
    header_json_str = raw_bytes[8:total_header_bytes].decode("utf-8")
    header_dict = json.loads(header_json_str)
    return header_dict, total_header_bytes


def parse_local_safetensors_header(file_path: Union[str, Path]) -> Tuple[Dict[str, Any], int]:
    """Read only the header portion of a local Safetensors file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Invalid Safetensors file {path}: less than 8 bytes")
        header_length = struct.unpack("<Q", header_len_bytes)[0]
        header_json_bytes = f.read(header_length)
        if len(header_json_bytes) < header_length:
            raise ValueError(f"Incomplete header in Safetensors file {path}")
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


def fetch_remote_bytes(
    url: str,
    byte_start: int,
    byte_end: int,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> bytes:
    """Fetch exact byte range from remote URL with redirect range preservation."""
    opener = urllib.request.build_opener(RedirectRangeHandler)
    req_headers = {
        "User-Agent": "PocketTitan/0.1.0",
        "Range": f"bytes={byte_start}-{byte_end}",
    }
    if headers:
        req_headers.update(headers)
        
    req = urllib.request.Request(url, headers=req_headers)
    with opener.open(req, timeout=timeout) as resp:
        return resp.read()


def parse_safetensors_header(
    source: Union[str, Path],
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], int]:
    """Universal header parser supporting local paths or HTTP(S) URLs."""
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        return parse_remote_safetensors_header(source_str, headers=headers)
    return parse_local_safetensors_header(source)
