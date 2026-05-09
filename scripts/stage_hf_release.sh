#!/usr/bin/env bash
# Stage the HuggingFace dataset release for `boson-ai/proactbench-data`
# into ./release/, using relative symlinks so we never duplicate files.
#
# After running this, `release/` contains everything you upload to HF:
#   huggingface-cli upload boson-ai/proactbench-data ./release . --repo-type=dataset
#
# Idempotent: re-running rebuilds the symlinks and regenerates croissant.json.
# README.md is treated as authored content and is NOT touched if present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REL_DIR="$REPO_ROOT/release"
DATASET_DIR="$REPO_ROOT/dataset"

if [ ! -d "$DATASET_DIR" ]; then
  echo "error: $DATASET_DIR not found" >&2
  exit 1
fi

mkdir -p "$REL_DIR"

# Wipe previous symlinks and generated artifacts (but keep README.md if hand-edited).
find "$REL_DIR" -mindepth 1 -maxdepth 1 -type l -delete
rm -f "$REL_DIR/croissant.json" "$REL_DIR/LICENSE"

# --- Symlinks to existing data (relative, so the tree is portable) ----------

# Top-level JSONL data files.
for f in \
    final_dialogues.jsonl \
    blueprints.jsonl \
    validated_blueprints.jsonl \
    tasks.jsonl \
    selected_tasks.jsonl \
    validation_results.jsonl
do
  ln -sf "../dataset/$f" "$REL_DIR/$f"
done

# Whole subdirectories.
ln -sf "../dataset/eval" "$REL_DIR/eval"
ln -sf "../dataset/human_eval" "$REL_DIR/human_eval"

# Datasheet and citation.
ln -sf "../dataset/DATASHEET.md" "$REL_DIR/DATASHEET.md"
ln -sf "../CITATION.cff" "$REL_DIR/CITATION.cff"

# --- LICENSE (CC-BY-4.0) ---------------------------------------------------
# Fetched from creativecommons.org so we ship the canonical text.
curl -fsSL https://creativecommons.org/licenses/by/4.0/legalcode.txt \
  -o "$REL_DIR/LICENSE"

# --- Croissant metadata: patched copy of dataset/metadata.json -------------
# Patches: license -> CC-BY-4.0, url -> HF dataset, distribution paths stay
# (release/ keeps the same relative layout as dataset/).

python3 <<PY
import json, pathlib

repo = pathlib.Path("${REPO_ROOT}")
src  = repo / "dataset" / "metadata.json"
dst  = repo / "release" / "croissant.json"

m = json.loads(src.read_text())

m["license"] = "https://creativecommons.org/licenses/by/4.0/"
m["url"]     = "https://huggingface.co/datasets/boson-ai/proactbench-data"

# Surface the new license in any distribution entries that mention licensing.
for d in m.get("distribution", []):
    if isinstance(d, dict) and "license" in d:
        d["license"] = "https://creativecommons.org/licenses/by/4.0/"

dst.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {dst.relative_to(repo)} ({dst.stat().st_size} bytes)")
PY

echo
echo "Staged $REL_DIR:"
ls -la "$REL_DIR"
echo
echo "Next:"
echo "  1. Review release/README.md (dataset card)."
echo "  2. Update CITATION.cff with named authors before upload (currently 'Anonymous')."
echo "  3. Upload:"
echo "       huggingface-cli upload boson-ai/proactbench-data $REL_DIR . --repo-type=dataset"
