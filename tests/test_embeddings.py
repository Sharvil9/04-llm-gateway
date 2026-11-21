"""Synthetic artifacts and fake CPU encoders; no downloads or real model loads."""

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm_gateway.embeddings as embeddings
from llm_gateway.embeddings import (
    ArtifactPin,
    BoundedEmbedder,
    EmbeddingArtifactError,
    EmbeddingInputError,
    LocalEmbedder,
    ModelSpec,
    fingerprint_for_model,
    verify_model_files,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_model(monkeypatch):
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="embedding-test-", dir=artifacts) as directory:
        path = Path(directory)
        files, records = {}, {}
        for name in embeddings.MODELS["l6"].files:
            data = f"synthetic:{name}".encode()
            destination = path / name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            files[name] = ArtifactPin(len(data), sha256=digest)
            records[name] = {"sha256": digest, "bytes": len(data)}
        spec = ModelSpec("test/synthetic-encoder", "a" * 40, 4, files)
        monkeypatch.setitem(embeddings.MODELS, "synthetic", spec)
        manifest = {
            "schema_version": 1,
            "model": "synthetic",
            "model_id": spec.model_id,
            "revision": spec.revision,
            "files": records,
        }
        (path / "manifest.json").write_text(json.dumps(manifest))
        yield path, spec, manifest


def test_valid_synthetic_artifacts_match_independent_source_pins(synthetic_model):
    path, spec, manifest = synthetic_model
    assert verify_model_files(path) == (spec, manifest)


def test_pure_fingerprint_uses_verified_files_and_runtime_versions(synthetic_model, monkeypatch):
    path, _, manifest = synthetic_model
    monkeypatch.setattr(embeddings.importlib.metadata, "version", lambda _name: "test-v1")
    before = fingerprint_for_model(path)
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    assert fingerprint_for_model(path) == before
    monkeypatch.setattr(embeddings.importlib.metadata, "version", lambda _name: "test-v2")
    assert fingerprint_for_model(path) != before


def test_rewritten_manifest_cannot_authorize_changed_model_bytes(synthetic_model):
    path, _, manifest = synthetic_model
    filename = "onnx/model_quantized.onnx"
    old = (path / filename).read_bytes()
    changed = b"X" + old[1:]
    (path / filename).write_bytes(changed)
    manifest["files"][filename]["sha256"] = hashlib.sha256(changed).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingArtifactError, match="source pin"):
        verify_model_files(path)


def test_git_blob_pin_binds_small_file_even_if_manifest_is_rewritten(synthetic_model):
    path, spec, manifest = synthetic_model
    name = "tokenizer.json"
    data = (path / name).read_bytes()
    git_blob = hashlib.sha1(
        f"blob {len(data)}\0".encode() + data, usedforsecurity=False
    ).hexdigest()
    spec.files[name] = ArtifactPin(len(data), git_blob=git_blob)
    assert verify_model_files(path) == (spec, manifest)
    changed = b"X" + data[1:]
    (path / name).write_bytes(changed)
    manifest["files"][name]["sha256"] = hashlib.sha256(changed).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingArtifactError, match="source pin"):
        verify_model_files(path)


@pytest.mark.parametrize("change", ["revision", "missing_file", "symlink"])
def test_invalid_or_redirected_local_artifacts_are_refused(synthetic_model, change):
    path, _, manifest = synthetic_model
    if change == "revision":
        manifest["revision"] = "b" * 40
        (path / "manifest.json").write_text(json.dumps(manifest))
    else:
        target = path / "tokenizer.json"
        data = target.read_bytes()
        target.unlink()
        if change == "symlink":
            alternate = path / "alternate.json"
            alternate.write_bytes(data)
            target.symlink_to(alternate)
    with pytest.raises(EmbeddingArtifactError):
        verify_model_files(path)


def test_importing_embedding_support_does_not_import_optional_inference_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import llm_gateway.embeddings; "
            "assert not {'numpy', 'onnxruntime', 'tokenizers'}.intersection(sys.modules)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


class FakeEncoder:
    fingerprint = "fake-encoder-v1"
    max_tokens = 256

    def __init__(self, *, fail_blocked=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.fail_blocked = fail_blocked

    def encode(self, text):
        self.calls.append((text, threading.get_ident()))
        if text == "blocked":
            self.started.set()
            if not self.release.wait(timeout=3):
                raise RuntimeError("Synthetic encoder was not released")
            if self.fail_blocked:
                raise EmbeddingInputError("Synthetic blocked failure")
        if text == "invalid":
            raise EmbeddingInputError("Synthetic invalid input")
        return [1.0, 0.0]


@pytest.fixture
def completed_executor(monkeypatch):
    """Return completed worker futures before their callbacks can run."""

    def install(worker):
        loop = asyncio.get_running_loop()
        original = loop.run_in_executor
        futures = []

        def submit(executor, job, *args):
            if executor is not worker._executor:
                return original(executor, job, *args)
            future = loop.create_future()
            try:
                future.set_result(job(*args))
            except Exception as exc:
                future.set_exception(exc)
            futures.append(future)
            return future

        monkeypatch.setattr(loop, "run_in_executor", submit)
        return futures

    return install


@pytest.mark.parametrize("operation", ["encode", "match"])
@pytest.mark.parametrize("first_fails", [False, True], ids=["success", "failure"])
async def test_completed_job_releases_capacity_before_caller_resumes(
    completed_executor, operation, first_fails
):
    vector = [1.0] + [0.0] * 383

    def encode(text):
        if text == "invalid":
            raise EmbeddingInputError("Synthetic invalid input")
        return vector

    worker = BoundedEmbedder(SimpleNamespace(fingerprint="fake-v1", max_tokens=256, encode=encode))
    futures = completed_executor(worker)
    if operation == "match":
        np = pytest.importorskip("numpy")
        expected = (np.asarray(vector, dtype="<f4").tobytes(), None, 0.0)
    else:
        expected = vector

    async def request(text):
        if operation == "match":
            return await worker.match(text, [], 0.9)
        return await worker.encode(text)

    try:
        if first_fails:
            with pytest.raises(EmbeddingInputError, match="Synthetic invalid input"):
                await request("invalid")
        else:
            assert await request("valid") == expected
        # No loop yield: the first future's completion callback is still queued.
        assert await request("next") == expected
        assert len(futures) == 2 and all(future.done() for future in futures)
        assert worker._future is None
    finally:
        await worker.close()


@pytest.mark.parametrize("first_fails", [False, True], ids=["success", "failure"])
async def test_delayed_old_completion_callback_preserves_new_job(monkeypatch, first_fails):
    worker = BoundedEmbedder(FakeEncoder())
    loop = asyncio.get_running_loop()
    original = loop.run_in_executor
    old, current, observed = loop.create_future(), loop.create_future(), loop.create_future()
    if first_fails:
        old.set_exception(EmbeddingInputError("Synthetic old failure"))
    else:
        old.set_result([1.0, 0.0])
    submissions = iter([old, current])

    def submit(executor, job, *args):
        if executor is worker._executor:
            return next(submissions)
        return original(executor, job, *args)

    monkeypatch.setattr(loop, "run_in_executor", submit)

    async def consecutive_requests():
        if first_fails:
            with pytest.raises(EmbeddingInputError, match="Synthetic old failure"):
                await worker.encode("invalid")
        else:
            assert await worker.encode("valid") == [1.0, 0.0]
        # Observe after the worker's old callback, while the new request awaits.
        old.add_done_callback(lambda _done: observed.set_result(worker._future))
        return await worker.encode("new")

    caller = asyncio.create_task(consecutive_requests())
    try:
        assert await asyncio.wait_for(observed, 1) is current
        assert await worker.encode("would-queue") is None
        assert not caller.done() and not current.done()
        current.set_result([1.0, 0.0])
        assert await caller == [1.0, 0.0]
        assert worker._future is None
    finally:
        if not current.done():
            current.set_result([1.0, 0.0])
        if not caller.done():
            caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        await worker.close()


async def test_busy_worker_has_no_waiting_queue_and_runs_outside_event_loop():
    fake = FakeEncoder()
    worker = BoundedEmbedder(fake)
    first = asyncio.create_task(worker.encode("blocked"))
    try:
        assert await asyncio.to_thread(fake.started.wait, 1)
        assert (
            await asyncio.gather(*[worker.encode("would-queue") for _ in range(30)]) == [None] * 30
        )
        assert len(fake.calls) == 1
        assert fake.calls[0][1] != threading.get_ident()
        assert worker.fingerprint == fake.fingerprint and worker.max_tokens == fake.max_tokens
        fake.release.set()
        assert await first == [1.0, 0.0]
        assert await worker.encode("next") == [1.0, 0.0]
        assert len(fake.calls) == 2
    finally:
        fake.release.set()
        await asyncio.gather(first, return_exceptions=True)
        await worker.close()


@pytest.mark.parametrize("fail_blocked", [False, True], ids=["success", "failure"])
async def test_caller_cancellation_does_not_release_running_worker_capacity(
    monkeypatch, fail_blocked
):
    fake = FakeEncoder(fail_blocked=fail_blocked)
    worker = BoundedEmbedder(fake)
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    notifications = []
    loop.set_exception_handler(lambda _loop, context: notifications.append(context))
    first = asyncio.create_task(worker.encode("blocked"))
    try:
        assert await asyncio.to_thread(fake.started.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await worker.encode("would-queue") is None
        assert await worker.match("would-queue", [], 0.9) is None
        assert len(fake.calls) == 1
        pending = worker._future
        assert not pending.done()
        consumed = []
        original_exception = pending.exception

        def exception():
            result = original_exception()
            consumed.append(result)
            return result

        monkeypatch.setattr(pending, "exception", exception)
        # Observe callback completion without awaiting/consuming the abandoned job.
        observed = asyncio.get_running_loop().create_future()
        pending.add_done_callback(lambda _done: observed.set_result(None))
        fake.release.set()
        await asyncio.wait_for(observed, timeout=1)
        assert worker._future is None
        assert consumed
        if fail_blocked:
            assert all(isinstance(error, EmbeddingInputError) for error in consumed)
        else:
            assert all(error is None for error in consumed)
        # Python 3.14 can explicitly report a shielded job's failure even after
        # consumption. Reject unhandled-future notifications or unrelated errors.
        for context in notifications:
            assert fail_blocked
            assert context["message"] == "EmbeddingInputError exception in shielded future"
            assert context["future"] is pending and context["exception"] in consumed
        assert await worker.encode("after-completion") == [1.0, 0.0]
    finally:
        try:
            fake.release.set()
            await worker.close()
        finally:
            loop.set_exception_handler(previous_handler)


async def test_worker_errors_propagate_and_capacity_recovers():
    worker = BoundedEmbedder(FakeEncoder())
    try:
        with pytest.raises(EmbeddingInputError):
            await worker.encode("invalid")
        assert await worker.encode("valid") == [1.0, 0.0]
    finally:
        await worker.close()


async def test_shutdown_waits_for_running_job_even_when_shutdown_caller_is_cancelled():
    fake, worker = FakeEncoder(), None
    worker = BoundedEmbedder(fake)
    first = asyncio.create_task(worker.encode("blocked"))
    closer = None
    try:
        assert await asyncio.to_thread(fake.started.wait, 1)
        closer = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        assert await worker.encode("after-close") is None
        closer.cancel()
        await asyncio.sleep(0)
        assert not closer.done()
        fake.release.set()
        assert await first == [1.0, 0.0]
        with pytest.raises(asyncio.CancelledError):
            await closer
        await worker.close()
    finally:
        fake.release.set()
        await asyncio.gather(first, return_exceptions=True)
        if closer is not None:
            await asyncio.gather(closer, return_exceptions=True)
        await worker.close()


def fake_local(np, *, token_count=4, hidden=None):
    local = LocalEmbedder.__new__(LocalEmbedder)
    local.max_tokens, local._np = 4, np
    tokens = SimpleNamespace(
        ids=list(range(token_count)),
        attention_mask=[1, 1] + [0] * (token_count - 2),
        type_ids=[0] * token_count,
    )
    local._tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: tokens)
    local._input_names = {"input_ids", "attention_mask", "token_type_ids"}
    if hidden is None:
        hidden = np.zeros((1, token_count, 384), dtype=np.float32)
        hidden[0, 0, 0], hidden[0, 1, 1], hidden[0, 2:, 2] = 1, 1, 100
    local._session = SimpleNamespace(run=lambda _outputs, _inputs: [hidden])
    return local


def test_masked_mean_pooling_normalizes_and_ignores_masked_tokens():
    np = pytest.importorskip("numpy")
    vector = fake_local(np).encode("synthetic input")
    assert vector.dtype == np.float32 and vector.shape == (384,)
    assert np.allclose(vector[:3], [2**-0.5, 2**-0.5, 0])
    assert np.isclose(np.linalg.norm(vector), 1)


def test_overlength_input_including_special_tokens_is_rejected_before_inference():
    np = pytest.importorskip("numpy")
    local = fake_local(np, token_count=5)
    assert local.token_count("synthetic input") == 5
    with pytest.raises(EmbeddingInputError, match="token limit"):
        local.encode("synthetic input")


@pytest.mark.parametrize("text", [None, "", "  ", "x" * 16001])
def test_invalid_embedding_input_is_rejected_without_tokenization(text):
    local = LocalEmbedder.__new__(LocalEmbedder)
    with pytest.raises(EmbeddingInputError):
        local.encode(text)


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_invalid_model_vectors_are_rejected(value):
    np = pytest.importorskip("numpy")
    local = fake_local(np, hidden=np.full((1, 4, 384), value, dtype=np.float32))
    with pytest.raises(EmbeddingArtifactError):
        local.encode("synthetic input")


async def test_match_normalizes_vectors_and_returns_only_threshold_qualified_candidate():
    np = pytest.importorskip("numpy")
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 2
    fake = SimpleNamespace(fingerprint="fake-v1", max_tokens=256, encode=lambda _text: vector)
    worker = BoundedEmbedder(fake)
    try:
        orthogonal = np.zeros(384, dtype="<f4")
        orthogonal[1] = 1
        candidates = [
            (b"malformed", b"short"),
            (b"orthogonal", orthogonal.tobytes()),
            (b"answer", vector.astype("<f4").tobytes()),
        ]
        encoded, key, score = await worker.match("question", candidates, 0.99)
        assert key == b"answer" and score == pytest.approx(1.0)
        assert np.allclose(np.frombuffer(encoded, dtype="<f4"), vector / 2)
        _, key, score = await worker.match("question", [(b"orthogonal", orthogonal.tobytes())], 0.9)
        assert key is None and score == pytest.approx(0)
        assert (await worker.match("question", [], 0.9))[1:] == (None, 0.0)
    finally:
        await worker.close()
