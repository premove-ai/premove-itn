# Inference artifact release

The public model source of truth is the Hugging Face model repository
`premove-ai/premove-itn`, revision `v0.1.0`. The release is an
inference-only export of the frozen structured-value 20k production
checkpoint. It contains no optimizer, scheduler, training counters, or
training data.

Published release:

- Repository: <https://huggingface.co/premove-ai/premove-itn>
- Python package: <https://pypi.org/project/premove-itn/>
- GitHub release: <https://github.com/premove-ai/premove-itn/releases/tag/v0.1.0>
- Immutable tag: `v0.1.0`
- Hub commit: `80bda5e2e1fe9542aa628597090242df57c1a157`
- Remote model size and SHA-256 verified against the accepted local artifact.

The maintained source for the public Hub landing page is
[`docs/model-card.md`](model-card.md). Update that file and review the Hub
change as a documentation-only pull request. Updating the card on `main` must
not move or replace the immutable `v0.1.0` tag.

The organization was renamed from `premove-itn` to `premove-ai` after this
release. The immutable `v0.1.0/provenance.json` therefore retains the original
`premove-itn/premove-itn-contextual` repository value. The package accepts that
historical value only for this frozen release.

## Build the local artifact

Run this from the repository root. The command refuses a checkpoint other than
the one named by `data/models/production.json` and verifies its frozen digest
before reading it.

```bash
uv sync --all-groups
mkdir -p artifacts
uv run python benchmarks/export_inference_artifact.py \
  --output artifacts/premove-itn-v0.1.0
```

The output is ignored by Git because `model.safetensors` is large. Keep the
local folder until the Hub upload and remote verification are complete.

## Run the acceptance gate

This compares the old production checkpoint and the inference artifact through
the same candidate builder and decoder. It performs no training.

```bash
uv run python benchmarks/verify_inference_artifact.py
```

The gate must report all of the following:

- `1500/1500` frozen VoiceAgent predictions identical.
- Identical tensor-state digests.
- Zero mismatch records.

The repository also has a separate development-only normalization regression
fixture. It is not a model-quality result or a publication gate.

The JSON and Markdown result are written under
`eval/voice_agent_itn/results/inference-artifact-v0.1.0/`.

## Authenticate and publish

Hugging Face requires an authenticated account before upload. Set a write
token in the environment or authenticate with the official CLI. Do not commit
the token.

```bash
# Or set HF_TOKEN in the environment. Never commit the token.
uv run hf auth login
uv run python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
repo_id = "premove-ai/premove-itn"
folder = "artifacts/premove-itn-v0.1.0"
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder,
    commit_message="Publish inference-only contextual model v0.1.0",
)
api.create_tag(
    repo_id=repo_id,
    repo_type="model",
    tag="v0.1.0",
    tag_message="Immutable first inference artifact",
)
print(api.model_info(repo_id, revision="v0.1.0").sha)
PY
```

Upload the complete folder in one call. The Hub upload API supports resumable
large-folder uploads. Do not move or overwrite the `v0.1.0` tag after this
step. Publish later changes under a new version.

## Verify the remote revision

After upload, download the tagged snapshot into a disposable directory and
compare its model digest with local `provenance.json`:

```bash
uv run python - <<'PY'
from pathlib import Path
import hashlib
from huggingface_hub import snapshot_download

local = Path("artifacts/premove-itn-v0.1.0")
remote = Path(snapshot_download(
    "premove-ai/premove-itn",
    revision="v0.1.0",
    allow_patterns=["model.safetensors", "config.json", "provenance.json", "base_config.json", "tokenizer/*", "README.md"],
))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

expected = __import__("json").loads((local / "provenance.json").read_text())["artifact_sha256"]
actual = sha256(remote / "model.safetensors")
assert actual == expected, (expected, actual)
print("remote model.safetensors matches local artifact", actual)
PY
```

The remote file hash match plus the local `1500/1500` gate is the release
acceptance evidence. The loader verifies the same digest whenever
`PremoveITN.from_pretrained()` loads the public release or a local downloaded
snapshot:

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained(revision="v0.1.0")
itn.normalize("call me at four thirty")  # "call me at 04:30"
```

## Licensing

The project code is MIT licensed. The artifact references
`microsoft/deberta-v3-large` at the pinned revision and includes the tokenizer
files generated from that revision. Its model card lists the base model under
the MIT license. Keep the attribution in the model card and comply with the
base model's terms when redistributing the artifact.
