#!/usr/bin/env python3
"""Zero-shot cell embedding with scGPT — one representation, no scoring.

Runs scGPT's own `scgpt.tasks.embed_data` (the reference-mapping tutorial's
core call) on an annotated AnnData and writes the resulting cell embedding.
Scoring happens in the metric collector, so this entrypoint stays a predictor.

    --data.h5ad        annotated AnnData with gene symbols and a label column
    --model.manifest   manifest pointing at a scGPT checkpoint directory in the
                       cache (args.json + best_model.pt + vocab.json)
"""

import argparse
import json
from pathlib import Path

from _data import load_cells, output_path, save_embedding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data.h5ad", dest="h5ad", required=True)
    p.add_argument("--model.manifest", dest="manifest", required=True)
    p.add_argument("--cells", type=int, default=200, help="cells to subsample")
    p.add_argument("--label_col", default="annot", help="obs column with cell types")
    p.add_argument("--max_length", type=int, default=1200, help="genes per cell")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", action="append", default=[], help="[id=]path")
    p.add_argument("--output_dir")
    p.add_argument("--name")
    args, _ = p.parse_known_args()

    import multiprocessing

    # scGPT defines its Dataset class inside embed_data, and Python 3.14 defaults
    # to the forkserver start method, which cannot pickle a local class
    multiprocessing.set_start_method("fork", force=True)

    from scgpt.tasks import embed_data

    manifest = json.loads(Path(args.manifest).read_text())
    adata, labels, dropped = load_cells(args.h5ad, args.cells, args.label_col)
    adata.var["gene_name"] = adata.var_names

    # embed_data filters adata to genes present in the vocab and returns that copy,
    # so the embedding lands on the returned object, not on ours
    embedded = embed_data(
        adata,
        Path(manifest["snapshot"]),  # args.json + best_model.pt + vocab.json
        gene_col="gene_name",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device="cpu",
        use_fast_transformer=False,  # flash-attn is a GPU-only optional extra
        # return_new_adata=True would hit AnnData(dtype=...), dropped in anndata 0.11
    )

    save_embedding(
        output_path(args.output, args.output_dir),
        f"scGPT ctx{args.max_length}",
        embedded.obsm["X_scGPT"],
        labels,
        dropped,
    )


if __name__ == "__main__":
    main()
