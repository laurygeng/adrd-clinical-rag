#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(description="Pull one or more Hugging Face models into a local cache directory.")
    p.add_argument(
        "models",
        nargs="+",
        help="One or more Hugging Face model IDs, e.g. BioMistral/BioMistral-7B-DARE or mistralai/Mistral-7B-Instruct-v0.3",
    )
    p.add_argument(
        "--base-dir",
        default="./downloaded_models",
        help="Base directory for downloaded models (default: ./downloaded_models)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
        help="HF token. If omitted, uses HUGGINGFACE_HUB_TOKEN or HF_TOKEN from env.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if target directory already exists and is non-empty.",
    )
    p.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional allow pattern(s) forwarded to snapshot_download. Can be repeated.",
    )
    p.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional ignore pattern(s) forwarded to snapshot_download. Can be repeated.",
    )
    return p.parse_args()


def safe_local_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def pull_one_model(model_id: str, base_dir: Path, token: str = None, force: bool = False, allow_patterns=None, ignore_patterns=None):
    local_dir = base_dir / safe_local_name(model_id)

    if is_nonempty_dir(local_dir) and not force:
        print(f"[SKIP] {model_id} already exists at: {local_dir}")
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PULL] Downloading {model_id} -> {local_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )

    print(f"[DONE] {model_id} -> {local_dir}")
    return str(local_dir)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base directory: {base_dir}")
    if args.token:
        print("HF token: detected")
    else:
        print("HF token: not set (public models only / lower rate limits)")

    for model_id in args.models:
        try:
            pull_one_model(
                model_id=model_id,
                base_dir=base_dir,
                token=args.token,
                force=args.force,
                allow_patterns=args.allow_pattern,
                ignore_patterns=args.ignore_pattern,
            )
        except Exception as e:
            print(f"[ERROR] Failed to pull {model_id}: {e}")


if __name__ == "__main__":
    main()
