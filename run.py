#!/usr/bin/env python3
"""omni-huggingface: fetch a Hugging Face dataset or model into a benchmark output.

Everything lands in the shared HF cache first (snapshot_download), then the
requested files are materialised at the paths omnibenchmark asks for via
`--output` (API 0.7, see design 008-named-outputs):

    --output PATH              single-output stage
    --output <id>=PATH         one per declared output otherwise

The output *suffix* decides what gets written:

    .json   a manifest: repo, revision, commit sha, cache path, file list
    .zip    every downloaded file, stored uncompressed (kind: zip)
    other   the single downloaded file, or the one matching the basename

Unknown flags are ignored, so the module can also sit in a non-entrypoint
stage and be handed inputs it does not care about.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def parse_outputs(values):
    """['id=path', ...] or ['path'] -> {id: Path}."""
    outputs = {}
    for value in values:
        oid, sep, path = value.partition("=")
        outputs[oid if sep else "default"] = Path(path if sep else oid)
    return outputs


def snapshot_files(snapshot: Path):
    """Payload files in a snapshot dir, skipping hub bookkeeping (.cache/, dotfiles)."""
    return sorted(
        p
        for p in snapshot.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(snapshot).parts)
    )


def materialize(snapshot: Path, outputs, meta):
    files = snapshot_files(snapshot)
    if not files:
        raise SystemExit(f"error: nothing downloaded into {snapshot}")
    for oid, dest in outputs.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".json":
            rel = [str(f.relative_to(snapshot)) for f in files]
            dest.write_text(json.dumps({**meta, "files": rel}, indent=2) + "\n")
        elif dest.suffix == ".zip":
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
                for f in files:
                    z.write(f, f.relative_to(snapshot))
        else:
            src = next(
                (f for f in files if f.name == dest.name),
                files[0] if len(files) == 1 else None,
            )
            if src is None:
                raise SystemExit(
                    f"error: output {oid} ({dest.name}) matches none of the "
                    f"downloaded files {[f.name for f in files]} — narrow the "
                    f"download with --files, or use a .zip/.json output"
                )
            shutil.copyfile(src, dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="HF repo id, e.g. tdc/scGPT")
    p.add_argument("--repo_type", default="model", choices=["model", "dataset"])
    p.add_argument("--revision", help="branch, tag or commit sha (default: main)")
    p.add_argument("--files", help="comma-separated globs to restrict the download")
    p.add_argument("--cache_dir", help="HF cache dir (default: HF_HOME / ~/.cache)")
    p.add_argument("--output", action="append", default=[], help="[id=]path")
    p.add_argument("--output_dir", help="fallback when no --output is given")
    p.add_argument("--name", help="unused, accepted for API compatibility")
    args, _ = p.parse_known_args()

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            args.repo,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=args.files.split(",") if args.files else None,
            cache_dir=args.cache_dir,
        )
    )
    meta = {
        "repo": args.repo,
        "repo_type": args.repo_type,
        "revision": args.revision or "main",
        "sha": snapshot.name,
        "snapshot": str(snapshot),
    }

    outputs = parse_outputs(args.output)
    if not outputs:
        # pre-0.7 API: no --output flags, mirror the snapshot into --output_dir.
        out = Path(args.output_dir or ".")
        outputs = {f.name: out / f.name for f in snapshot_files(snapshot)}
    materialize(snapshot, outputs, meta)

    # ponytail: no model instantiation — that drags in torch + the scGPT package
    # for something only the consuming module knows how to do. It gets the
    # snapshot path from the manifest and loads the weights itself.


if __name__ == "__main__":
    main()
