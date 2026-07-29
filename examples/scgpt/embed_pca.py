#!/usr/bin/env python3
"""PCA baseline — the ordinary scanpy pipeline, same cells, no model.

Exists so the collector has something to compare scGPT against: normalise,
log1p, PCA. Any flags it does not know (the model manifest, say) are ignored.

    --data_h5ad   annotated AnnData with gene symbols and a label column
"""

import argparse

from _data import load_cells, output_path, save_embedding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_h5ad", dest="h5ad", required=True)
    p.add_argument("--cells", type=int, default=200, help="cells to subsample")
    p.add_argument("--label_col", default="annot", help="obs column with cell types")
    p.add_argument("--components", type=int, default=50, help="principal components")
    p.add_argument("--output", action="append", default=[], help="[id=]path")
    p.add_argument("--output_dir")
    p.add_argument("--name")
    args, _ = p.parse_known_args()

    import scanpy as sc

    adata, labels, dropped = load_cells(args.h5ad, args.cells, args.label_col)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=args.components)

    save_embedding(
        output_path(args.output, args.output_dir),
        f"PCA({args.components})",
        adata.obsm["X_pca"],
        labels,
        dropped,
    )


if __name__ == "__main__":
    main()
