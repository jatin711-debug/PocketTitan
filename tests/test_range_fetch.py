"""Range fetches must resume, not truncate.

A 52 GiB download over a slow link will have its socket closed early at least
once. When that happened the read loop ended without error, a short buffer came
back, and the failure surfaced three hours later as

    shape '[17408, 5120]' is invalid for input of size 47516081

which names neither the tensor nor the cause. These tests serve deliberately
truncated responses from a real local HTTP server.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from pockettitan.config import TensorAddress, TruncatedTensorError
from pockettitan.metadata.safetensors_header import (
    IncompleteRangeRead,
    RangeResponseError,
    fetch_remote_bytes,
)
from pockettitan.streaming.reader import RemoteTensorSliceReader

PAYLOAD = bytes((i * 7 + 3) % 256 for i in range(64 * 1024))


class _Server:
    """Serves byte ranges of PAYLOAD, cutting the first `truncate_first` responses short."""

    def __init__(self, truncate_first: int = 0, cut_to: int = 1024):
        self.truncate_first = truncate_first
        self.cut_to = cut_to
        self.requests: list = []
        payload, state = PAYLOAD, self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                header = self.headers.get("Range", "")
                start, _, end = header.replace("bytes=", "").partition("-")
                start = int(start)
                end = int(end) if end else len(payload) - 1
                body = payload[start : end + 1]
                state.requests.append((start, end))

                truncated = len(state.requests) <= state.truncate_first
                sent = body[: state.cut_to] if truncated else body

                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                # Advertise the full range: a truncating proxy does not announce it.
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(sent)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/shard.safetensors"


def test_a_complete_range_is_returned_verbatim():
    with _Server() as server:
        got = fetch_remote_bytes(server.url, 0, len(PAYLOAD) - 1, timeout=10)
    assert got == PAYLOAD


def test_a_server_that_ignores_range_fails_closed():
    payload = PAYLOAD

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        with pytest.raises(RangeResponseError, match="possible full-shard download"):
            fetch_remote_bytes(f"http://{host}:{port}/shard", 0, 15)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_malformed_content_range_fails_closed():
    payload = PAYLOAD

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(206)
            self.send_header("Content-Range", "not-a-range")
            self.end_headers()
            self.wfile.write(payload[:16])

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        secret = "hf_do_not_log_this_token"
        with pytest.raises(RangeResponseError, match="Malformed") as caught:
            fetch_remote_bytes(
                f"http://{host}:{port}/shard",
                0,
                15,
                headers={"Authorization": f"Bearer {secret}"},
            )
        assert secret not in str(caught.value)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_truncated_response_resumes_from_where_it_stopped():
    """The retry must request only the missing suffix, not the whole range again."""
    with _Server(truncate_first=1, cut_to=1024) as server:
        got = fetch_remote_bytes(
            server.url, 0, len(PAYLOAD) - 1, timeout=10, backoff_seconds=0.01
        )
        assert got == PAYLOAD
        assert len(server.requests) == 2
        first, second = server.requests
        assert first[0] == 0
        assert second[0] == 1024, "the resume must start at the byte the first attempt reached"


def test_repeated_truncation_still_completes_within_the_retry_budget():
    with _Server(truncate_first=3, cut_to=4096) as server:
        got = fetch_remote_bytes(
            server.url, 0, len(PAYLOAD) - 1, timeout=10, backoff_seconds=0.01
        )
    assert got == PAYLOAD


def test_an_exhausted_retry_budget_raises_and_says_what_is_missing():
    with _Server(truncate_first=99, cut_to=512) as server:
        with pytest.raises(IncompleteRangeRead, match="of 65,536 bytes"):
            fetch_remote_bytes(
                server.url,
                0,
                len(PAYLOAD) - 1,
                timeout=10,
                max_attempts=3,
                backoff_seconds=0.01,
            )


def test_progress_is_reported_once_per_byte_not_once_per_attempt():
    """Resumed bytes must not be double counted, or the progress bar overshoots."""
    seen = []
    with _Server(truncate_first=1, cut_to=1024) as server:
        fetch_remote_bytes(
            server.url,
            0,
            len(PAYLOAD) - 1,
            timeout=10,
            backoff_seconds=0.01,
            chunk_callback=lambda n, total: seen.append((n, total)),
        )
    assert sum(n for n, _ in seen) == len(PAYLOAD)
    assert {total for _, total in seen} == {len(PAYLOAD)}


def test_range_fetch_can_be_cancelled_before_opening_a_request():
    cancel = threading.Event()
    cancel.set()
    with _Server() as server:
        with pytest.raises(InterruptedError, match="Cancelled range fetch"):
            fetch_remote_bytes(server.url, 0, len(PAYLOAD) - 1, cancel_event=cancel)
        assert server.requests == []


# --------------------------------------------------------------------------- #
# The message the failure produces
# --------------------------------------------------------------------------- #


def test_short_payload_names_the_tensor_not_a_shape():
    with pytest.raises(TruncatedTensorError, match="transfer was cut short"):
        RemoteTensorSliceReader._bytes_to_tensor(b"\x00" * 100, "BF16", [17408, 5120])


def test_a_complete_payload_still_converts():
    values = np.arange(6, dtype=np.uint16)
    tensor = RemoteTensorSliceReader._bytes_to_tensor(values.tobytes(), "BF16", [2, 3])
    assert tuple(tensor.shape) == (2, 3)


def test_an_indivisible_row_stride_is_refused():
    """An integer-division stride that leaves a remainder shifts every later row."""
    address = TensorAddress(
        name="w",
        shard="s.safetensors",
        dtype="BF16",
        shape=[3, 5],
        byte_start=0,
        byte_end=31,
        size_bytes=31,
        num_params=15,
    )
    reader = RemoteTensorSliceReader("org/model")
    with pytest.raises(TruncatedTensorError, match="do not divide into 3 rows"):
        reader.read_slice(address, 0, 1)
