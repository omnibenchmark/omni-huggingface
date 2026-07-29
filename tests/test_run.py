#!/usr/bin/env python3
"""Self-check for the output-materialisation and report logic (no network)."""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples/scgpt"))
from scgpt_metrics import report  # noqa: E402
from run import materialize, parse_outputs, snapshot_files  # noqa: E402


def test():
    assert parse_outputs(["/a/b.h5ad"]) == {"default": Path("/a/b.h5ad")}
    assert parse_outputs(["m.weights=/a/b.json", "m.raw=/a/c.zip"]) == {
        "m.weights": Path("/a/b.json"),
        "m.raw": Path("/a/c.zip"),
    }
    # ob 0.6 passes a workflow-relative path but runs us from the module checkout:
    # the file must still land in --output_dir, not under the module
    assert parse_outputs(["data/D1/.abc/D1.h5ad"], "/out/data/D1/.abc") == {
        "default": Path("/out/data/D1/.abc/D1.h5ad")
    }
    assert parse_outputs(["m.raw=data/D1/.abc/D1.zip"], "/out/data/D1/.abc") == {
        "m.raw": Path("/out/data/D1/.abc/D1.zip")
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

        meta = {"repo": "tdc/scGPT", "sha": "abc"}
        materialize(
            snap,
            {
                "m.manifest": out / "scgpt.json",
                "m.bundle": out / "scgpt.zip",
                "m.data": out / "pbmc.h5ad",
            },
            meta,
        )
        manifest = json.loads((out / "scgpt.json").read_text())
        assert manifest["repo"] == "tdc/scGPT"
        assert manifest["files"] == ["config.json", "pbmc.h5ad"]
        assert zipfile.ZipFile(out / "scgpt.zip").namelist() == [
            "config.json",
            "pbmc.h5ad",
        ]
        assert (out / "pbmc.h5ad").read_text() == "data"

        # ambiguous plain-file output: several candidates, no basename match
        try:
            materialize(snap, {"m.data": out / "nope.h5ad"}, meta)
        except SystemExit as e:
            assert "matches none" in str(e)
        else:
            raise AssertionError("expected a SystemExit for an unmatched output")

    # the report tabulates one row per representation, in insertion order
    rows = {
        "scGPT zero-shot": {"dim": 512, "knn_acc": 0.855, "macro_f1": 0.448, "ari": 0.410},
        "PCA(50) baseline": {"dim": 50, "knn_acc": 0.860, "macro_f1": 0.451, "ari": 0.474},
    }
    text = report("zero-shot cell embedding", rows)
    assert text.splitlines()[-2].split()[0] == "scGPT"
    assert "0.474" in text

    print("OK")


if __name__ == "__main__":
    test()
