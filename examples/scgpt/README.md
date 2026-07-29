# Example: scGPT zero-shot cell embedding

Runs scGPT's reference-mapping tutorial as a benchmark. Three stages, all served
by the omni-huggingface module: pull an annotated PBMC h5ad (97 MB) and the
whole-human scGPT checkpoint (196 MB), embed the cells with scGPT's own
`scgpt.tasks.embed_data` — no fine-tuning — and score how separable the
annotated cell types are, against a PCA baseline on the same cells.

```sh
ob run benchmark examples/scgpt/benchmark.yaml
# or, without omnibenchmark:
bash examples/scgpt/run.sh out
```

```
zero-shot cell embedding — perturblab/scgpt-human
197 cells, 6 annotated types, 1200-gene context; 2 type(s) dropped as too rare to score: Treg, cDC

representation       dim   knn_acc  macro_f1     ari
scGPT zero-shot      512     0.863     0.600   0.522
PCA(50) baseline      50     0.868     0.600   0.561
```

The PCA baseline is the point: here the 512-dimensional zero-shot embedding does
not beat 50 principal components. CPU only, ~2 min per 200 cells at a 600-gene
context, ~4 min at 1200.

## Files

| File | Role |
|------|------|
| `benchmark.yaml` | The benchmark: data → model → metrics |
| `scgpt_metrics.py` | The `scgpt_metrics` entrypoint, declared in the module's `omnibenchmark.yaml` |
| `env.yml` | Conda env for the metrics stage |
| `pixi.toml` | Dev env for running it by hand (`pixi run install-scgpt` first) |
| `run.sh` | The same three steps without omnibenchmark |

## Caveats

- The PBMC shard stores expression pre-binned to 10 levels, not raw counts.
  scGPT re-bins into its own 51 bins and the binning is monotone per cell, so the
  ordering survives — but the input is coarser than the tutorial's.
- scGPT is installed `--no-deps`: its metadata pins `scvi-tools<1.0` and
  `orbax<0.1.8`, which the embedding path never imports.
- Cell types with fewer than 5 cells are dropped — 5-fold stratified CV needs
  them.
