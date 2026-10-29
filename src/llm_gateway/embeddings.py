"""Optional pinned CPU embeddings and a single bounded asynchronous worker.

Importing this module does not import NumPy, ONNX Runtime, or tokenizers. Model
files are never downloaded implicitly. Tokenization rejects overlength input;
it never silently truncates a question before semantic comparison.
"""

import asyncio
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class EmbeddingInputError(ValueError):
    """The input cannot safely be represented by this embedding model."""


class EmbeddingArtifactError(ValueError):
    """Model artifacts are absent, untrusted, or incompatible."""


@dataclass(frozen=True)
class ArtifactPin:
    size: int
    sha256: str | None = None
    git_blob: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    max_tokens: int
    files: Mapping[str, ArtifactPin]


# ONNX SHA256 and small-file Git blob IDs come from the pinned HF ?blobs=true
# revision metadata. SHA256 manifests are checked against these independent
# source pins, so rewriting a local manifest cannot authorize modified weights
# or tokenizer data. Small-file SHA256 pins can additionally be recorded after
# the downloader verifies their committed Git blob identities.
MODELS = {
    "l6": ModelSpec(
        "Xenova/all-MiniLM-L6-v2",
        "751bff37182d3f1213fa05d7196b954e230abad9",
        256,
        {
            "config.json": ArtifactPin(650, git_blob="72147e4ff4426ebedbfa2146c4a0999def51a313"),
            "tokenizer.json": ArtifactPin(
                711661, git_blob="c17ed520ed8438736732a54957a69306b8822215"
            ),
            "tokenizer_config.json": ArtifactPin(
                366, git_blob="37fca74771bc76a8e01178ce3a6055a0995f8093"
            ),
            "special_tokens_map.json": ArtifactPin(
                125, git_blob="a8b3208c2884c4efb86e49300fdd3dc877220cdf"
            ),
            "onnx/model_quantized.onnx": ArtifactPin(
                22972370, sha256="afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"
            ),
        },
    ),
    "l3": ModelSpec(
        "Xenova/paraphrase-MiniLM-L3-v2",
        "4b544e74dfc3256b2b56849ea5d7064fee1ac846",
        128,
        {
            "config.json": ArtifactPin(657, git_blob="fe45035abdece7de4993e59da54513f7d33a7a80"),
            "tokenizer.json": ArtifactPin(
                711649, git_blob="327401998c676bd2d7c625039113b0577ea49fe7"
            ),
            "tokenizer_config.json": ArtifactPin(
                366, git_blob="37fca74771bc76a8e01178ce3a6055a0995f8093"
            ),
            "special_tokens_map.json": ArtifactPin(
                125, git_blob="a8b3208c2884c4efb86e49300fdd3dc877220cdf"
            ),
            "onnx/model_quantized.onnx": ArtifactPin(
                17452106, sha256="b190f50dd46296b9895ae8f274c3455762d08610f8788f4a9bd15019f4f7160c"
            ),
        },
    ),
}


def artifact_hashes(path: Path, pin: ArtifactPin) -> dict[str, str | int]:
    """Verify source-pinned content and return a SHA256 manifest record."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size != pin.size:
        raise EmbeddingArtifactError("Missing or unexpected embedding artifact size")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {pin.size}\0".encode("ascii"), usedforsecurity=False)
    with path.open("rb") as source:
        while chunk := source.read(65536):
            sha256.update(chunk)
            git_blob.update(chunk)
    digest = sha256.hexdigest()
    if (
        (not pin.sha256 and not pin.git_blob)
        or (pin.sha256 and digest != pin.sha256)
        or (pin.git_blob and git_blob.hexdigest() != pin.git_blob)
    ):
        raise EmbeddingArtifactError("Embedding artifact does not match its source pin")
    return {"sha256": digest, "bytes": pin.size}


def verify_model_files(model_dir: Path) -> tuple[ModelSpec, dict]:
    """Verify a local manifest and every allowed file without loading a model."""
    directory = Path(model_dir).resolve()
    manifest_file = directory / "manifest.json"
    try:
        if manifest_file.is_symlink() or manifest_file.stat().st_size > 16384:
            raise EmbeddingArtifactError("Invalid embedding manifest")
        manifest = json.loads(manifest_file.read_text())
        spec = MODELS[manifest["model"]]
        if (
            manifest["schema_version"] != 1
            or manifest["model_id"] != spec.model_id
            or manifest["revision"] != spec.revision
            or set(manifest["files"]) != set(spec.files)
        ):
            raise EmbeddingArtifactError("Embedding manifest does not match the pinned registry")
        for name, pin in spec.files.items():
            path = directory / name
            if not path.resolve().is_relative_to(directory):
                raise EmbeddingArtifactError("Embedding artifact escapes its model directory")
            if artifact_hashes(path, pin) != manifest["files"][name]:
                raise EmbeddingArtifactError("Embedding manifest checksum mismatch")
        return spec, manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeError, RecursionError):
        raise EmbeddingArtifactError("Missing or invalid pinned embedding model files") from None


def _fingerprint(spec: ModelSpec, manifest: dict) -> str:
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in ["numpy", "onnxruntime", "tokenizers"]
        }
    except importlib.metadata.PackageNotFoundError:
        raise EmbeddingArtifactError("Install the optional semantic dependencies first") from None
    identity = {
        "pipeline": "onnx-masked-mean-float32-l2-v1",
        "manifest": manifest,
        "max_tokens": spec.max_tokens,
        "threads": 1,
        "allow_spinning": False,
        "packages": versions,
    }
    return (
        "embedding-v1:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def fingerprint_for_model(model_dir: Path) -> str:
    """Verify artifacts and identify their pipeline without loading ONNX/tokenizers."""
    return _fingerprint(*verify_model_files(model_dir))


class LocalEmbedder:
    """Quantized BERT, masked mean pooling, float32 L2 normalization on CPU."""

    def __init__(self, model_dir: Path):
        spec, manifest = verify_model_files(model_dir)
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            raise EmbeddingArtifactError(
                "Install the optional semantic dependencies first"
            ) from None
        self.max_tokens = spec.max_tokens
        self._np = np
        self._tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(Path(model_dir) / "onnx/model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {node.name for node in self._session.get_inputs()}
        if not self._input_names <= {"input_ids", "attention_mask", "token_type_ids"}:
            raise EmbeddingArtifactError("Unsupported embedding model input contract")
        self.fingerprint = _fingerprint(spec, manifest)

    def _tokenize(self, text: str):
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise EmbeddingInputError("Embedding input must be a nonempty bounded string")
        return self._tokenizer.encode(text, add_special_tokens=True)

    def token_count(self, text: str) -> int:
        """Count all tokens including special tokens, without truncation."""
        return len(self._tokenize(text).ids)

    def encode(self, text: str) -> "np.ndarray":
        tokens = self._tokenize(text)
        if len(tokens.ids) > self.max_tokens:
            raise EmbeddingInputError("Embedding input exceeds the model token limit")
        np = self._np
        inputs = {
            "input_ids": np.asarray([tokens.ids], dtype=np.int64),
            "attention_mask": np.asarray([tokens.attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([tokens.type_ids], dtype=np.int64),
        }
        outputs = self._session.run(None, {name: inputs[name] for name in self._input_names})
        hidden = [value for value in outputs if value.shape == (1, len(tokens.ids), 384)]
        if len(hidden) != 1:
            raise EmbeddingArtifactError("Unsupported embedding model output contract")
        mask = inputs["attention_mask"][..., None].astype(np.float32)
        pooled = (np.asarray(hidden[0], dtype=np.float32) * mask).sum(axis=1)[0] / mask.sum()
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm) or norm <= 0 or not np.isfinite(pooled).all():
            raise EmbeddingArtifactError("Invalid embedding vector")
        return np.asarray(pooled / norm, dtype=np.float32)


class Encoder(Protocol):
    fingerprint: str
    max_tokens: int

    def encode(self, text: str) -> Any: ...


class BoundedEmbedder:
    """One worker, zero waiting queue, capacity held until actual completion.

    Cancellation stops the caller's wait, not its running CPU job. Shutdown waits
    for that job before joining the executor. The instance belongs to one loop.
    """

    def __init__(self, encoder: Encoder):
        self._encoder = encoder
        self.fingerprint = encoder.fingerprint
        self.max_tokens = encoder.max_tokens
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gateway-embedding")
        self._loop = None
        self._future = None
        self._closed = False
        self._shutdown = None

    def _owning_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("Embedding worker belongs to another event loop")
        self._loop = loop
        return loop

    async def _run(self, job: Callable):
        loop = self._owning_loop()
        if self._closed or self._future is not None:
            return None
        future = loop.run_in_executor(self._executor, job)
        self._future = future

        def finished(done):
            if self._future is done:
                self._future = None
            if not done.cancelled():
                done.exception()  # Consume errors even when the caller was cancelled.

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    async def encode(self, text: str) -> "np.ndarray | None":
        return await self._run(lambda: self._encoder.encode(text))

    async def match(
        self, text: str, candidates: list[tuple[bytes, bytes]], threshold: float
    ) -> tuple[bytes, bytes | None, float] | None:
        snapshot = tuple(candidates)
        return await self._run(lambda: self._match(text, snapshot, threshold))

    def _match(self, text: str, candidates: tuple, threshold: float):
        import numpy as np

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            or len(candidates) > 128
        ):
            raise EmbeddingInputError("Invalid semantic lookup bounds")
        vector = np.asarray(self._encoder.encode(text), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if (
            vector.shape != (384,)
            or not np.isfinite(vector).all()
            or not math.isfinite(norm)
            or norm <= 0
        ):
            raise EmbeddingInputError("Invalid query embedding")
        vector = np.asarray(vector / norm, dtype="<f4")
        rows, keys = [], []
        for key, encoded in candidates:
            if (
                not isinstance(key, bytes)
                or not isinstance(encoded, bytes)
                or len(encoded) != vector.nbytes
            ):
                continue
            candidate = np.frombuffer(encoded, dtype="<f4")
            norm = float(np.linalg.norm(candidate))
            if np.isfinite(candidate).all() and math.isfinite(norm) and norm > 0:
                rows.append(candidate / norm)
                keys.append(key)
        if not rows:
            return vector.tobytes(), None, 0.0
        scores = np.clip(np.stack(rows) @ vector, -1.0, 1.0)
        best = int(np.argmax(scores))
        score = float(scores[best])
        return vector.tobytes(), keys[best] if score >= threshold else None, score

    async def close(self) -> None:
        loop = self._owning_loop()
        self._closed = True
        if self._shutdown is None:
            self._shutdown = loop.run_in_executor(
                None, lambda: self._executor.shutdown(wait=True, cancel_futures=False)
            )
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._shutdown)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
