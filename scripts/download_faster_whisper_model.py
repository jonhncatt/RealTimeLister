from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ALLOWED_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a faster-whisper model directory that can be copied to an offline machine."
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Model size or HF repo id. Examples: small, medium, Systran/faster-whisper-small",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory. Example: C:/models/faster-whisper-small",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token.",
    )
    return parser.parse_args()


def resolve_repo_id(model: str) -> str:
    if "/" in model:
        return model
    return f"Systran/faster-whisper-{model}"


def validate_model_dir(path: Path) -> None:
    required = ["config.json", "model.bin", "tokenizer.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"Downloaded directory is incomplete, missing: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = resolve_repo_id(args.model)
    print(f"Downloading {repo_id} to {output_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=ALLOWED_PATTERNS,
        token=args.hf_token,
    )
    validate_model_dir(output_dir)

    print("Download complete.")
    print(f"Use this on the offline machine with RT_ASR_MODEL_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
