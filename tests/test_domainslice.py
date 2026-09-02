"""DomainSlice V0: exact expert ranges, durable cache, and restart reuse."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pockettitan.config import ModelMetadata, TensorAddress
from pockettitan.domainslice import (
    CacheBudgetError,
    CompositeWeightStore,
    ModelRevision,
    PocketTitanPageStore,
    RemoteHuggingFaceStore,
    WeightPageID,
)
from pockettitan.metadata.tensor_index import TensorAddressTable

MIB = 1024 * 1024
PROJECTION_BYTES = 4 * MIB
EXPERT_BYTES = 3 * PROJECTION_BYTES


class _RangeServer:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.requests: list[tuple[int, int, str | None]] = []
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                value = self.headers.get("Range", "")
                start_text, _, end_text = value.removeprefix("bytes=").partition("-")
                start, end = int(start_text), int(end_text)
                state.requests.append((start, end, self.headers.get("Authorization")))
                body = state.payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(state.payload)}")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
        return f"http://{host}:{port}/model.safetensors"


def _fixture() -> tuple[bytes, TensorAddressTable, list[bytes]]:
    projections = [b"g" * PROJECTION_BYTES, b"u" * PROJECTION_BYTES, b"d" * PROJECTION_BYTES]
    starts = [128, 128 + PROJECTION_BYTES + 37, 128 + 2 * PROJECTION_BYTES + 79]
    payload = bytearray(starts[-1] + PROJECTION_BYTES + 16)
    names = ["gate_proj", "up_proj", "down_proj"]
    tensors = {}
    for expert_id in (7, 8):
        for name, start, data in zip(names, starts, projections):
            payload[start : start + len(data)] = data
            tensor_name = f"model.layers.9.mlp.experts.{expert_id}.{name}.weight"
            tensors[tensor_name] = TensorAddress(
                name=tensor_name,
                shard="model.safetensors",
                dtype="BF16",
                shape=[1024, 2048] if name != "down_proj" else [2048, 1024],
                byte_start=start,
                byte_end=start + len(data),
                num_params=2_097_152,
                size_bytes=len(data),
            )
    metadata = ModelMetadata(
        architecture="OlmoeForCausalLM",
        num_hidden_layers=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=1024,
        vocab_size=50304,
        total_params=6_919_161_856,
        is_moe=True,
        num_experts=64,
        num_experts_per_tok=8,
        expert_intermediate_size=1024,
        source_dtype="bfloat16",
        shards=["model.safetensors"],
        tensors=tensors,
    )
    return bytes(payload), TensorAddressTable(metadata), projections


def _stores(tmp_path, server, table, *, workers=3, max_cache=50 * MIB, remote_cls=None):
    revision = ModelRevision(repo_id="test/olmoe", commit_sha="a" * 40)
    cls = remote_cls or RemoteHuggingFaceStore
    remote = cls(
        revision,
        address_table=table,
        max_workers=workers,
        url_resolver=lambda _shard: server.url,
    )
    local = PocketTitanPageStore(tmp_path / "cache", max_cache_bytes=max_cache)
    composite = CompositeWeightStore(local, remote, download_workers=workers)
    return revision, composite


def test_olmoe_expert_resolves_to_three_exact_bf16_ranges(tmp_path):
    payload, table, _projections = _fixture()
    with _RangeServer(payload) as server:
        revision, store = _stores(tmp_path, server, table)
        try:
            descriptor = store.resolve(WeightPageID.expert(revision, 9, 7))
            assert descriptor.expected_bytes == 12_582_912
            assert [item.projection for item in descriptor.source_slices] == [
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            assert [item.size_bytes for item in descriptor.source_slices] == [
                PROJECTION_BYTES,
                PROJECTION_BYTES,
                PROJECTION_BYTES,
            ]
            assert descriptor.output_layout.payload_bytes == EXPERT_BYTES
        finally:
            store.close()


def test_arbitrary_bf16_tensor_uses_the_same_immutable_page_cache(tmp_path):
    payload, table, _projections = _fixture()
    tensor_name = "model.layers.9.mlp.experts.7.gate_proj.weight"
    with _RangeServer(payload) as server:
        revision, store = _stores(tmp_path, server, table)
        page_id = WeightPageID.tensor(revision, tensor_name)
        try:
            descriptor = store.resolve(page_id)
            assert descriptor.page_id.page_kind == "tensor"
            assert descriptor.expected_bytes == PROJECTION_BYTES
            assert descriptor.output_layout.projection("tensor").shape == [1024, 2048]
            first = store.materialize(page_id)
            assert first.bytes_fetched == PROJECTION_BYTES
            store.release(first)
            second = store.materialize(page_id)
            assert second.cache_hit is True
            assert second.bytes_fetched == 0
            store.release(second)
        finally:
            store.close()


def test_first_fault_fetches_exact_payload_and_second_fault_is_local(tmp_path):
    payload, table, projections = _fixture()
    with _RangeServer(payload) as server:
        revision, store = _stores(tmp_path, server, table)
        page_id = WeightPageID.expert(revision, 9, 7)
        try:
            first = store.materialize(page_id)
            assert first.cache_hit is False
            assert first.bytes_fetched == EXPERT_BYTES
            assert first.path.read_bytes() == b"".join(projections)
            requested = sum(end - start + 1 for start, end, _auth in server.requests)
            assert requested == EXPERT_BYTES
            request_count = len(server.requests)
            store.release(first)

            second = store.materialize(page_id)
            assert second.cache_hit is True
            assert second.bytes_fetched == 0
            assert len(server.requests) == request_count
            stats = store.stats()
            assert stats.cache_hits == 1
            assert stats.cache_misses == 1
            assert stats.remote_payload_bytes == EXPERT_BYTES
            store.release(second)
        finally:
            store.close()


def test_interrupted_fault_reuses_completed_projection(tmp_path):
    payload, table, _projections = _fixture()

    class InterruptOnce(RemoteHuggingFaceStore):
        failed = False

        def fetch_slice_to_file(self, source_slice, destination, **kwargs):
            if source_slice.projection == "up_proj" and not self.failed:
                self.failed = True
                raise RuntimeError("injected interruption")
            return super().fetch_slice_to_file(source_slice, destination, **kwargs)

    with _RangeServer(payload) as server:
        revision, store = _stores(
            tmp_path, server, table, workers=1, remote_cls=InterruptOnce
        )
        page_id = WeightPageID.expert(revision, 9, 7)
        try:
            with pytest.raises(RuntimeError, match="injected interruption"):
                store.materialize(page_id)
            before_retry = len(server.requests)
            handle = store.materialize(page_id)
            assert handle.bytes_resumed >= PROJECTION_BYTES
            assert handle.bytes_fetched <= 2 * PROJECTION_BYTES
            assert len(server.requests) - before_retry <= 2
            store.release(handle)
        finally:
            store.close()


def test_mid_projection_interruption_resumes_only_verified_prefix(tmp_path):
    payload, table, _projections = _fixture()
    cancel = threading.Event()
    revision = ModelRevision(repo_id="test/olmoe", commit_sha="a" * 40)
    with _RangeServer(payload) as server:
        remote = RemoteHuggingFaceStore(
            revision,
            address_table=table,
            max_workers=1,
            chunk_size=MIB,
            url_resolver=lambda _shard: server.url,
        )
        local = PocketTitanPageStore(tmp_path / "cache", max_cache_bytes=50 * MIB)
        store = CompositeWeightStore(local, remote, download_workers=1)
        page_id = WeightPageID.expert(revision, 9, 7)

        def stop_after_first_chunk(_stage, _item, _count, _total):
            cancel.set()

        try:
            with pytest.raises(InterruptedError):
                store.materialize(page_id, progress=stop_after_first_chunk, cancel_event=cancel)
            cancel.clear()
            handle = store.materialize(page_id)
            assert handle.bytes_resumed == MIB
            assert handle.bytes_fetched == EXPERT_BYTES - MIB
            assert store.stats().remote_payload_bytes == EXPERT_BYTES
            store.release(handle)
        finally:
            store.close()


def test_corrupt_page_is_discarded_and_refetched(tmp_path):
    payload, table, _projections = _fixture()
    with _RangeServer(payload) as server:
        revision, store = _stores(tmp_path, server, table)
        page_id = WeightPageID.expert(revision, 9, 7)
        try:
            first = store.materialize(page_id)
            store.release(first)
            with first.path.open("r+b") as stream:
                stream.seek(123)
                stream.write(b"corrupt")
            previous_requests = len(server.requests)
            repaired = store.materialize(page_id)
            assert repaired.cache_hit is False
            assert len(server.requests) == previous_requests + 3
            assert store.stats().corruptions == 1
            store.release(repaired)
        finally:
            store.close()


def test_revision_changes_page_identity_and_too_small_budget_fails_before_fetch(tmp_path):
    payload, table, _projections = _fixture()
    first = ModelRevision(repo_id="test/olmoe", commit_sha="a" * 40)
    second = ModelRevision(repo_id="test/olmoe", commit_sha="b" * 40)
    assert WeightPageID.expert(first, 9, 7).cache_key != WeightPageID.expert(
        second, 9, 7
    ).cache_key

    with _RangeServer(payload) as server:
        revision, store = _stores(tmp_path, server, table, max_cache=EXPERT_BYTES - 1)
        try:
            with pytest.raises(CacheBudgetError, match="cache budget"):
                store.materialize(WeightPageID.expert(revision, 9, 7))
            assert server.requests == []
        finally:
            store.close()


def test_cache_never_evicts_a_leased_page(tmp_path):
    payload, table, _projections = _fixture()
    with _RangeServer(payload) as server:
        revision, store = _stores(tmp_path, server, table, max_cache=EXPERT_BYTES)
        first_id = WeightPageID.expert(revision, 9, 7)
        second_id = WeightPageID.expert(revision, 9, 8)
        try:
            first = store.materialize(first_id)
            requests_before = len(server.requests)
            with pytest.raises(CacheBudgetError, match="currently in use"):
                store.materialize(second_id)
            assert len(server.requests) == requests_before
            assert first.path.exists()

            store.release(first)
            second = store.materialize(second_id)
            assert second.path.exists()
            assert not first.path.exists()
            assert store.stats().evictions == 1
            store.release(second)
        finally:
            store.close()
