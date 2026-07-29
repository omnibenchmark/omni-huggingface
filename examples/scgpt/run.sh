#!/usr/bin/env bash
# Standalone invocation of the two entrypoints, without omnibenchmark.
# Run from anywhere:
#   bash examples/scgpt/run.sh [outdir]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-out}"
mkdir -p "$OUT"

pixi() { command pixi run --manifest-path "$ROOT/pixi.toml" -e scgpt "$@"; }

pixi python "$ROOT/run.py" \
  --repo Chtholly17/PBMC_ATLAS \
  --repo_type dataset \
  --files "pbmc_atlas_rna_binning_10_test.h5ad" \
  --output "data.h5ad=$OUT/pbmc_atlas.h5ad"

pixi python "$ROOT/run.py" \
  --repo perturblab/scgpt-human \
  --repo_type model \
  --revision 571a0445d68fa48381f863ff75dd4f6d0eae3dfc \
  --files "args.json,best_model.pt,vocab.json" \
  --output "model.manifest=$OUT/scgpt_human_model.json"

pixi python "$ROOT/examples/scgpt/scgpt_metrics.py" \
  --data.h5ad "$OUT/pbmc_atlas.h5ad" \
  --model.manifest "$OUT/scgpt_human_model.json" \
  --cells 200 \
  --max_length 1200 \
  --output "metrics.cell_type_separability=$OUT/cell_type_separability_metrics.json"

echo "Metrics written to $OUT/cell_type_separability_metrics.json"
