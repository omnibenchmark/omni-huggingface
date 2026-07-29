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

The download always lands in the shared HF cache. What the module then puts at
the path your benchmark declares depends on **the file extension of that path** —
that extension is how you tell it what you want:

```yaml
    outputs:
      - id: counts
        path: "{name}.h5ad"    # copies the downloaded .h5ad here
      - id: weights
        path: "{name}.json"    # writes a manifest; the weights stay in the cache
```

| You declare | The module writes |
|-------------|-------------------|
| `.json` | A manifest: repo, revision, commit sha, path of the cache snapshot, list of files |
| `.zip` | Every downloaded file in one uncompressed zip (declare `kind: zip` alongside) |
| any other suffix | One downloaded file copied into place — the one whose filename matches, or the only one if the download has a single file |

Reach for `.json` whenever the repo is big or multi-file. Model weights run to
hundreds of megabytes, and a plain output would copy them into every node's
directory; the manifest is a few hundred bytes pointing at the one cached copy:

```json
{ "repo": "perturblab/scgpt-human",
  "sha": "571a0445d68fa48381f863ff75dd4f6d0eae3dfc",
  "snapshot": "~/.cache/huggingface/hub/models--perturblab--scgpt-human/snapshots/571a044…",
  "files": ["args.json", "best_model.pt", "vocab.json"] }
```

A downstream module reads `snapshot` from it and loads the weights from the
cache — that is what [`examples/scgpt/`](examples/scgpt/) does.

Given no `--output` at all (pre-0.7 API), the snapshot is mirrored into
`--output_dir` under the original filenames.

## Development

```sh
pixi run check       # self-check, no network
pixi run export-env  # regenerate envs/omni-huggingface.yml
```

## Example

[`examples/scgpt/`](examples/scgpt/) is a worked benchmark built on this module.
It is self-contained — own conda env, own pixi manifest, no imports from
`run.py` — and takes nothing from the module but the three entrypoints
`omnibenchmark.yaml` declares on its behalf.
