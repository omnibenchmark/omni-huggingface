# omni-huggingface

Hugging Face Hub download module for [omnibenchmark](https://omnibenchmark.org) pipelines.
Fetches datasets or model weights into the shared HF cache, then materialises them
at the paths the benchmark declares (it needs [named outputs](https://github.com/omnibenchmark/omnibenchmark/pull/329), API 0.7).

## Usage

```yaml
modules:
  - id: weights
    repository:
      url: https://github.com/btraven00/omni-huggingface
      commit: main
    parameters:
      - repo: "perturblab/scgpt-human"
        repo_type: "model"
        files: "args.json,best_model.pt,vocab.json"
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `repo` | Hub repo id, e.g. `org/model-name` |
| `repo_type` | `model` (default) or `dataset` |
| `revision` | Branch, tag or commit sha (default: `main`) — pin it, `main` moves |
| `files` | Comma-separated globs to restrict the download, e.g. `*.h5ad` |
| `cache_dir` | HF cache location (default: `HF_HOME`, else `~/.cache/huggingface`) |

## Outputs

The suffix of each declared output decides what is written there:

| Suffix | Content |
|--------|---------|
| `.json` | Manifest: repo, revision, commit sha, cache snapshot path, file list |
| `.zip` | Every downloaded file, stored uncompressed (`kind: zip`) |
| anything else | The single downloaded file, or the one whose basename matches |

Big files stay in the HF cache: point a `.json` output at them and read
`snapshot` from the manifest downstream. Given no `--output` (pre-0.7 API), the
snapshot is mirrored into `--output_dir` under the original filenames.

## Development

```sh
pixi run check       # self-check, no network
pixi run export-env  # regenerate envs/omni-huggingface.yml
```

## Example

[`examples/scgpt/`](examples/scgpt/) is a worked benchmark built on this module.
It is self-contained — own conda env, own pixi manifest, no imports from
`run.py` — and takes nothing from the module but the `scgpt_metrics` entrypoint
that `omnibenchmark.yaml` declares on its behalf.
