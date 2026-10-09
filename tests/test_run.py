#!/usr/bin/env python3
"""Self-check for the module's output materialisation (no network)."""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run import materialize, mirror, parse_outputs, snapshot_files  # noqa: E402


def test():
    assert parse_outputs(["/a/b.h5ad"]) == {"default": Path("/a/b.h5ad")}
    assert parse_outputs(["m_weights=/a/b.json", "m_raw=/a/c.zip"]) == {
        "m_weights": Path("/a/b.json"),
        "m_raw": Path("/a/c.zip"),
    }
    # ob 0.6 passes a workflow-relative path but runs us from the module checkout:
    # the file must still land in --output_dir, not under the module
    assert parse_outputs(["data/D1/.abc/D1.h5ad"], "/out/data/D1/.abc") == {
        "default": Path("/out/data/D1/.abc/D1.h5ad")
    }
    assert parse_outputs(["m_raw=data/D1/.abc/D1.zip"], "/out/data/D1/.abc") == {
        "m_raw": Path("/out/data/D1/.abc/D1.zip")
    }
    assert parse_outputs(["/abs/D1.h5ad"], "/out") == {"default": Path("/abs/D1.h5ad")}

    with tempfile.TemporaryDirectory() as tmp:
        snap, out = Path(tmp) / "snap", Path(tmp) / "out"
        (snap / ".cache" / "huggingface").mkdir(parents=True)
        (snap / ".cache" / "huggingface" / "meta").write_text("bookkeeping")
        (snap / ".gitattributes").write_text("lfs")
        (snap / "config.json").write_text("{}")
        (snap / "pbmc.h5ad").write_text("data")
        assert [f.name for f in snapshot_files(snap)] == ["config.json", "pbmc.h5ad"]
        # a snapshot dir holds every cached file of its revision; --files keeps the asked-for ones
        (snap / "pbmc_50k.h5ad").write_text("other rung, fetched by an earlier node")
        assert [f.name for f in snapshot_files(snap, ["pbmc.h5ad"])] == ["pbmc.h5ad"]
        assert [f.name for f in snapshot_files(snap, ["*.json"])] == ["config.json"]
        only = Path(tmp) / "only"
        mirror(snap, only, ["pbmc.h5ad"])
        assert sorted(f.name for f in only.iterdir()) == ["pbmc.h5ad"]
        (snap / "pbmc_50k.h5ad").unlink()

        meta = {"repo": "org/model", "sha": "abc"}
        materialize(
            snap,
            {
                "m_manifest": out / "weights.json",
                "m_bundle": out / "weights.zip",
                "m_data": out / "pbmc.h5ad",
            },
            meta,
        )
        manifest = json.loads((out / "weights.json").read_text())
        assert manifest["repo"] == "org/model"
        assert manifest["files"] == ["config.json", "pbmc.h5ad"]
        assert zipfile.ZipFile(out / "weights.zip").namelist() == [
            "config.json",
            "pbmc.h5ad",
        ]
        assert (out / "pbmc.h5ad").read_text() == "data"

        # the pre-0.7 fallback mirrors verbatim: a downloaded *.json stays itself
        # and must not be rewritten into a manifest of itself
        mirror(snap, out / "mirrored")
        assert (out / "mirrored" / "config.json").read_text() == "{}"
        assert (out / "mirrored" / "pbmc.h5ad").read_text() == "data"
        assert not (out / "mirrored" / ".gitattributes").exists()

        # ambiguous plain-file output: several candidates, no basename match
        try:
            materialize(snap, {"m_data": out / "nope.h5ad"}, meta)
        except SystemExit as e:
            assert "matches none" in str(e)
        else:
            raise AssertionError("expected a SystemExit for an unmatched output")

    print("OK")


if __name__ == "__main__":
    test()
