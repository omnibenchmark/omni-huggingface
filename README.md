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
      - repo: "tdc/scGPT"
        repo_type: "model"
        files: "config.json,model.safetensors"
```

See [`examples/scgpt/benchmark.yaml`](examples/scgpt/benchmark.yaml) for a complete benchmark.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `repo` | Hub repo id, e.g. `tdc/scGPT`, `datavil/pbmc3k` |
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
```

## Citation

See `CITATION.cff`.
