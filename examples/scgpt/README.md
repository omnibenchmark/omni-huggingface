# Example: scGPT zero-shot cell embedding

Runs scGPT's reference-mapping tutorial as a benchmark: pull an annotated PBMC
h5ad (97 MB) and the whole-human scGPT checkpoint (196 MB), embed the cells with
scGPT's own `scgpt.tasks.embed_data` — no fine-tuning — and score how separable
the annotated cell types are.

Prediction, baseline and scoring are separate steps. Each embedding module emits
a representation and nothing else; the metric collector scores all of them in one
place. Adding a competing embedding means adding a module to the `embedding`
stage — the collector picks it up unchanged.

```
data ──► model ──► embedding ┬── scgpt (ctx 600)  ─┐
                             ├── scgpt (ctx 1200) ─┼─► metric collector
                             └── pca (50 comps)   ─┘
```

```sh
ob run benchmark examples/scgpt/benchmark.yaml
# or, without omnibenchmark:
bash examples/scgpt/run.sh out
```

```
cell-type separability — 3 representations; types too rare to score: Treg, cDC

representation        cells   dim   knn_acc  macro_f1     ari
PCA(50)                 197    50     0.868     0.600   0.561
scGPT ctx1200           197   512     0.868     0.604   0.519
scGPT ctx600            197   512     0.868     0.634   0.515
```

The PCA baseline is the point: here the 512-dimensional zero-shot embedding does
not beat 50 principal components. CPU only, ~2 min per 200 cells at a 600-gene
context, ~4 min at 1200.

## Files

| File | Role |
|------|------|
| `benchmark.yaml` | The benchmark: data → model → embedding → collector |
| `embed_scgpt.py` | `embed_scgpt` entrypoint — scGPT representation |
| `embed_pca.py` | `embed_pca` entrypoint — PCA baseline representation |
| `collect_metrics.py` | `collect_metrics` entrypoint — the metric collector |
| `_data.py` | Shared cell loading and output helpers |
| `env.yml` | Conda env for the embedding and metrics steps |
| `pixi.toml` | Dev env for running it by hand (`pixi run install-scgpt` first) |
| `run.sh` | The same steps without omnibenchmark |

Each representation is stored as an `.npz` carrying its own labels, so the
collector scores each one independently — no row alignment across modules.

## Caveats

- The PBMC shard stores expression pre-binned to 10 levels, not raw counts.
  scGPT re-bins into its own 51 bins and the binning is monotone per cell, so the
  ordering survives — but the input is coarser than the tutorial's.
- scGPT is installed `--no-deps`: its metadata pins `scvi-tools<1.0` and
  `orbax<0.1.8`, which the embedding path never imports.
- Cell types with fewer than 5 cells are dropped — 5-fold stratified CV needs
  them.
- Collector nodes get no `--output` flags, so `collect_metrics.py` writes a fixed
  `metrics.json` into `--output_dir`; the benchmark declares the matching path.
- The scGPT rows move by ~0.02 between runs while the PCA row is bit-identical:
  torch's CPU reduction order varies with thread scheduling, and a handful of the
  197 cells flip their k-NN neighbourhood. Set `torch.set_num_threads(1)` if you
  need reproducible digits.
