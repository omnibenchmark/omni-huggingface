#!/usr/bin/env python3
"""Metric collector: score every cell embedding produced upstream.

Gathers the `.npz` representations from all modules of the embedding stage —
scGPT at each context length, the PCA baseline — and scores each one on its own
labels, so nothing has to be row-aligned across representations:

    knn_acc   accuracy of a cross-validated k-NN cell-type classifier
    macro_f1  macro-averaged F1 of the same classifier (rare types count too)
    ari       adjusted Rand index of k-means against the annotated labels

Collector nodes receive `--<input_id> path1 path2 ...` and no `--output` flags,
so the metrics table is written to `--output_dir` under a fixed name.
"""

import argparse
import json
from pathlib import Path

OUTPUT_NAME = "metrics.json"


def score(features, labels, neighbors, seed=0):
    """k-NN cell-type accuracy plus k-means agreement for one representation."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, f1_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    predicted = cross_val_predict(
        KNeighborsClassifier(n_neighbors=neighbors), features, labels, cv=5
    )
    clusters = KMeans(
        n_clusters=len(set(labels)), n_init=10, random_state=seed
    ).fit_predict(features)
    return {
        "cells": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "knn_acc": float((predicted == labels).mean()),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "ari": float(adjusted_rand_score(labels, clusters)),
    }


def report(header, rows):
    lines = [
        header,
        "",
        f"{'representation':<20}{'cells':>7}{'dim':>6}{'knn_acc':>10}"
        f"{'macro_f1':>10}{'ari':>8}",
    ]
    for r in rows:
        lines.append(
            f"{r['representation']:<20}{r['cells']:>7}{r['dim']:>6}"
            f"{r['knn_acc']:>10.3f}{r['macro_f1']:>10.3f}{r['ari']:>8.3f}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedding_cells", dest="embeddings", nargs="+", required=True)
    p.add_argument("--neighbors", type=int, default=15, help="k for the k-NN classifier")
    p.add_argument("--output_dir")
    p.add_argument("--name")
    args, _ = p.parse_known_args()

    import numpy as np

    rows, dropped = [], set()
    for path in sorted(args.embeddings):
        loaded = np.load(path, allow_pickle=True)
        labels = loaded["labels"].astype(str)
        dropped.update(loaded["dropped"].tolist())
        rows.append(
            {
                "representation": str(loaded["name"]),
                "source": Path(path).name,
                **score(loaded["X"], labels, args.neighbors),
            }
        )
    rows.sort(key=lambda r: -r["knn_acc"])

    header = f"cell-type separability — {len(rows)} representations"
    if dropped:
        header += f"; types too rare to score: {', '.join(sorted(dropped))}"
    print(report(header, rows))

    dest = Path(args.output_dir or ".") / OUTPUT_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"neighbors": args.neighbors, "representations": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
