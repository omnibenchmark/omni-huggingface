# omni-huggingface

Hugging Face Hub download module for [omnibenchmark](https://omnibenchmark.org) pipelines.
Fetches datasets or model weights into the shared HF cache, then materialises them
at the paths the benchmark declares ([named outputs](https://github.com/omnibenchmark/omnibenchmark/pull/329), API 0.7).

## Usage

```yaml
modules:
  - id: scgpt
    repository:
      url: https://github.com/btraven00/omni-huggingface
      commit: main
    parameters:
      - repo: "perturblab/scgpt-human"
        repo_type: "model"
        files: "args.json,best_model.pt,vocab.json"
```

See [`examples/scgpt/`](examples/scgpt/) for a complete benchmark.

## Entrypoints

| Entrypoint | Script | Purpose |
|------------|--------|---------|
| `default` | `run.py` | Download a dataset or model repo |
| `scgpt_metrics` | `examples/scgpt/scgpt_metrics.py` | Zero-shot cell embedding metrics for scGPT |

Select a non-default one with `entrypoint: scgpt_metrics` in the module block.
`envs/omni-huggingface.yml` is enough for `default`; the metrics entrypoint needs
`envs/omni-scgpt.yml`, which adds pytorch, scanpy, scikit-learn and scGPT itself.

## Example

[`examples/scgpt/benchmark.yaml`](examples/scgpt/benchmark.yaml) runs scGPT's
zero-shot reference-mapping tutorial as a benchmark: pull an annotated PBMC h5ad
(97 MB) and the whole-human scGPT checkpoint (196 MB), embed cells with scGPT's
own `scgpt.tasks.embed_data` — no fine-tuning — and score how separable the
annotated cell types are, against a PCA baseline on the same cells. One run per
`max_length` value:

```
ob run benchmark -b examples/scgpt/benchmark.yaml --local
# or, without omnibenchmark:
bash examples/scgpt/run.sh out

zero-shot cell embedding — perturblab/scgpt-human
197 cells, 6 annotated types, 600-gene context; 2 type(s) dropped as too rare to score: Treg, cDC

representation       dim   knn_acc  macro_f1     ari
scGPT zero-shot      512     0.863     0.597   0.511
PCA(50) baseline      50     0.868     0.600   0.561
```

The PCA baseline is there on purpose: on this data the 512-dimensional zero-shot
embedding does not beat 50 principal components, which is the kind of result the
benchmark exists to surface.

### Caveats

- The PBMC atlas shard stores expression pre-binned to 10 levels, not raw counts.
  scGPT re-bins into its own 51 bins, and the binning is monotone per cell, so the
  ordering survives — but the input is coarser than the tutorial's.
- scGPT is installed `--no-deps` (see `envs/omni-scgpt.yml`); its published
  metadata pins `scvi-tools<1.0` and `orbax<0.1.8`, which the embedding path
  never imports.
- CPU only, ~2 min per 200 cells at a 600-gene context.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `repo` | Hub repo id, e.g. `perturblab/scgpt-human`, `Chtholly17/PBMC_ATLAS` |
| `repo_type` | `model` (default) or `dataset` |
| `revision` | Branch, tag or commit sha (default: `main`) |
| `files` | Comma-separated globs to restrict the download, e.g. `*.h5ad` |
| `cache_dir` | HF cache location (default: `HF_HOME`, else `~/.cache/huggingface`) |

Pin `revision` to a commit sha for reproducible runs — `main` moves.

## Outputs

The suffix of each declared output decides what the module writes there:

| Suffix | Content |
|--------|---------|
| `.json` | Manifest: repo, revision, commit sha, cache snapshot path, file list |
| `.zip` | Every downloaded file, stored uncompressed (`kind: zip`) |
| anything else | The single downloaded file, or the one whose basename matches |

Weights stay in the HF cache; use a `.json` output and read `snapshot` from the
manifest to load them downstream. The module does not instantiate models — that
needs the framework only the consuming module knows about.

If the runtime passes no `--output` (pre-0.7 API), the snapshot is mirrored into
`--output_dir` under the original filenames.

## Development

```sh
pixi run check       # self-check, no network
pixi run export-env  # regenerate envs/omni-huggingface.yml
pixi run -e scgpt install-scgpt  # scGPT itself, --no-deps
```

## Citation

See `CITATION.cff`.
