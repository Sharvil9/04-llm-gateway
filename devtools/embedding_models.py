"""Download only pinned embedding artifacts; never load or run a model.

Usage: python -m devtools.embedding_models --model l6 --output artifacts/embeddings/l6
Uses the inherited HTTPS proxy and CA configuration with certificate verification.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import httpx

from llm_gateway.embeddings import MODELS, EmbeddingArtifactError, artifact_hashes

ROOT = Path(__file__).resolve().parents[1]


def download(model: str, output: Path) -> dict:
    artifacts = ROOT / "artifacts"
    output = (ROOT / output).resolve()
    if not output.is_relative_to(artifacts) or output == artifacts:
        raise ValueError("Embedding output must be a directory below this repository's artifacts/")
    spec = MODELS[model]
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"Accept-Encoding": "identity"},
    ) as client:
        for name, pin in spec.files.items():
            target = output / name
            if not target.resolve().is_relative_to(output):
                raise ValueError("Embedding artifact path escapes its output directory")
            if target.exists():
                try:
                    records[name] = artifact_hashes(target, pin)
                    continue
                except EmbeddingArtifactError:
                    pass
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".download-", delete=False
                ) as destination:
                    temporary = Path(destination.name)
                    size, started = 0, time.monotonic()
                    url = f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{name}"
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        for chunk in response.iter_bytes(chunk_size=65536):
                            size += len(chunk)
                            if size > pin.size or time.monotonic() - started > 180:
                                raise EmbeddingArtifactError(
                                    "Embedding download exceeds its size/time bound"
                                )
                            destination.write(chunk)
                records[name] = artifact_hashes(temporary, pin)
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "model": model,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "files": records,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Embedding manifest must not be a symbolic link")
    temporary_manifest = output / ".manifest.json.tmp"
    if temporary_manifest.is_symlink():
        raise ValueError("Embedding manifest temporary path must not be a symbolic link")
    temporary_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return {"directory": str(output.relative_to(ROOT)), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts/embeddings") / args.model
    print(json.dumps(download(args.model, output), indent=2))


if __name__ == "__main__":
    main()
