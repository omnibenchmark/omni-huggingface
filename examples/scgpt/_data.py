"""Shared cell loading for the embedding entrypoints of this example."""

from pathlib import Path

MIN_CELLS_PER_TYPE = 5  # 5-fold stratified CV in the collector needs this many


def load_cells(h5ad, n_cells, label_col):
    """Read the h5ad, subsample deterministically, drop types too rare to score."""
    import scanpy as sc

    adata = sc.read_h5ad(h5ad)
    if n_cells < adata.n_obs:
        # same seed in every embedding module, so all representations describe
        # the same cells even though each module subsamples independently
        sc.pp.subsample(adata, n_obs=n_cells, random_state=0)
    counts = adata.obs[label_col].astype(str).value_counts()
    rare = sorted(counts[counts < MIN_CELLS_PER_TYPE].index)
    adata = adata[~adata.obs[label_col].astype(str).isin(rare)].copy()
    return adata, adata.obs[label_col].astype(str).to_numpy(), rare


def save_embedding(dest, name, embedding, labels, dropped):
    """Write one representation: the matrix plus the labels it is scored against."""
    import numpy as np

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        X=np.asarray(embedding, dtype="float32"),
        labels=np.asarray(labels, dtype=object),
        name=name,
        dropped=np.asarray(dropped, dtype=object),
    )
    print(f"{name}: {np.shape(embedding)} embedding -> {dest}")


def output_path(values, output_dir, default="embedding.npz"):
    """--output [id=]path -> where to write.

    Kept self-contained on purpose: this example must not import from the module
    it sits next to. omnibenchmark 0.6 passed snakemake's workflow-relative path
    while running us from the module checkout; resolving a relative path against
    --output_dir stays correct now that the runtime passes absolute paths.
    """
    raw = (values[0].partition("=")[2] or values[0]) if values else default
    path = Path(raw)
    return path if path.is_absolute() else Path(output_dir or ".") / path.name
