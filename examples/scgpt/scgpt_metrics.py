#!/usr/bin/env python3
"""Zero-shot cell embedding with scGPT, following the reference-mapping tutorial.

Cui et al., "scGPT: toward building a foundation model for single-cell
multi-omics using generative AI", Nature Methods 21:1470-1480 (2024),
doi:10.1038/s41592-024-02201-0. Runs scGPT's own `scgpt.tasks.embed_data` on an
annotated AnnData — no fine-tuning — then asks whether cell types are separable
in the resulting embedding, against a PCA baseline on the same cells.

Inputs (both produced by run.py):
    --model.manifest   manifest pointing at a scGPT checkpoint directory in the
                       cache (args.json + best_model.pt + vocab.json)
    --data.h5ad        annotated AnnData with gene symbols and a label column

Reported per representation:
    knn_acc   accuracy of a cross-validated k-NN cell-type classifier
    macro_f1  macro-averaged F1 of the same classifier (rare types count too)
    ari       adjusted Rand index of k-means against the annotated labels

Writes a metrics JSON at --output and prints a report to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for run.py
from run import parse_outputs  # noqa: E402


def score(features, labels, neighbors, seed=0):
    """k-NN cell-type accuracy plus k-means agreement for one representation."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, f1_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    predicted = cross_val_predict(
        KNeighborsClassifier(n_neighbors=neighbors), features, labels, cv=5
    )
    clusters = KMeans(n_clusters=len(set(labels)), n_init=10, random_state=seed).fit_predict(
        features
    )
    return {
        "dim": features.shape[1],
        "knn_acc": float((predicted == labels).mean()),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "ari": float(adjusted_rand_score(labels, clusters)),
    }


def report(header, rows):
    lines = [header, "", f"{'representation':<18}{'dim':>6}{'knn_acc':>10}{'macro_f1':>10}{'ari':>8}"]
    for name, m in rows.items():
        lines.append(
            f"{name:<18}{m['dim']:>6}{m['knn_acc']:>10.3f}{m['macro_f1']:>10.3f}{m['ari']:>8.3f}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model.manifest", dest="manifest", required=True)
    p.add_argument("--data.h5ad", dest="h5ad", required=True)
    p.add_argument("--cells", type=int, default=500, help="cells to subsample")
    p.add_argument("--label_col", default="annot", help="obs column with cell types")
    p.add_argument("--max_length", type=int, default=1200, help="genes per cell")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--neighbors", type=int, default=15, help="k for the k-NN classifier")
    p.add_argument("--output", action="append", default=[], help="[id=]path")
    p.add_argument("--output_dir")
    p.add_argument("--name")
    args, _ = p.parse_known_args()

    import multiprocessing

    # scGPT defines its Dataset class inside embed_data, and Python 3.14 defaults
    # to the forkserver start method, which cannot pickle a local class
    multiprocessing.set_start_method("fork", force=True)

    import numpy as np
    import scanpy as sc
    from scgpt.tasks import embed_data

    manifest = json.loads(Path(args.manifest).read_text())
    model_dir = Path(manifest["snapshot"])  # args.json + best_model.pt + vocab.json

    adata = sc.read_h5ad(args.h5ad)
    if args.cells < adata.n_obs:
        sc.pp.subsample(adata, n_obs=args.cells, random_state=0)
    adata.var["gene_name"] = adata.var_names
    # 5-fold stratified CV needs 5 cells per type; rarer ones cannot be scored
    counts = adata.obs[args.label_col].astype(str).value_counts()
    dropped = sorted(counts[counts < 5].index)
    adata = adata[~adata.obs[args.label_col].astype(str).isin(dropped)].copy()
    labels = adata.obs[args.label_col].astype(str).to_numpy()

    # embed_data filters adata to genes present in the vocab and returns that copy,
    # so the embedding lands on the returned object, not on ours
    embedded = embed_data(
        adata,
        model_dir,
        gene_col="gene_name",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device="cpu",
        use_fast_transformer=False,  # flash-attn is a GPU-only optional extra
        # return_new_adata=True would hit AnnData(dtype=...), dropped in anndata 0.11
    )
    embedding = embedded.obsm["X_scGPT"]

    # baseline: the ordinary scanpy pipeline on the same cells, so the question is
    # "does the foundation model beat PCA here", not "is the embedding non-random"
    baseline = adata.copy()
    sc.pp.normalize_total(baseline, target_sum=1e4)
    sc.pp.log1p(baseline)
    sc.pp.pca(baseline, n_comps=50)

    rows = {
        "scGPT zero-shot": score(np.asarray(embedding), labels, args.neighbors),
        "PCA(50) baseline": score(baseline.obsm["X_pca"], labels, args.neighbors),
    }

    header = (
        f"zero-shot cell embedding — {manifest['repo']}\n"
        f"{adata.n_obs} cells, {len(set(labels))} annotated types, "
        f"{args.max_length}-gene context"
        + (f"; {len(dropped)} type(s) dropped as too rare to score: {', '.join(dropped)}" if dropped else "")
    )
    print(report(header, rows))

    metrics = {
        "model": manifest["repo"],
        "revision": manifest["revision"],
        "sha": manifest["sha"],
        "cells": int(adata.n_obs),
        "label_col": args.label_col,
        "types": sorted(set(labels)),
        "dropped_types": dropped,
        "max_length": args.max_length,
        "representations": rows,
    }
    outputs = parse_outputs(args.output, args.output_dir)
    dest = next(iter(outputs.values()), Path(args.output_dir or ".") / "metrics.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
