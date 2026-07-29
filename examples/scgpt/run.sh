#!/usr/bin/env bash
# Standalone invocation of the two entrypoints, without omnibenchmark.
# Run from anywhere:
#   bash examples/scgpt/run.sh [outdir]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-out}"
mkdir -p "$OUT"

pixi() { command pixi run --manifest-path "$(dirname "$0")/pixi.toml" "$@"; }

pixi python "$ROOT/run.py" \
  --repo Chtholly17/PBMC_ATLAS \
  --repo_type dataset \
  --files "pbmc_atlas_rna_binning_10_test.h5ad" \
  --output "data_h5ad=$OUT/pbmc_atlas.h5ad"

pixi python "$ROOT/run.py" \
  --repo perturblab/scgpt-human \
  --repo_type model \
  --revision 571a0445d68fa48381f863ff75dd4f6d0eae3dfc \
  --files "args.json,best_model.pt,vocab.json" \
  --output "model_manifest=$OUT/scgpt_human_model.json"

pixi python "$ROOT/examples/scgpt/embed_scgpt.py" \
  --data_h5ad "$OUT/pbmc_atlas.h5ad" \
  --model_manifest "$OUT/scgpt_human_model.json" \
  --cells 200 --max_length 1200 \
  --output "embedding_cells=$OUT/scgpt.npz"

pixi python "$ROOT/examples/scgpt/embed_pca.py" \
  --data_h5ad "$OUT/pbmc_atlas.h5ad" \
  --cells 200 --components 50 \
  --output "embedding_cells=$OUT/pca.npz"

pixi python "$ROOT/examples/scgpt/collect_metrics.py" \
  --embedding_cells "$OUT/scgpt.npz" "$OUT/pca.npz" \
  --output_dir "$OUT"

echo "Metrics written to $OUT/metrics.json"
