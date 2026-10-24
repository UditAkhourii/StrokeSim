#!/usr/bin/env bash
# Fetch external data for ffsim. Idempotent: existing files are skipped.
# Run from anywhere; paths are resolved relative to the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fetch() {  # fetch URL DEST
  if [ -s "$2" ]; then echo "skip  $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "get   $2"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

# 1. Shiu et al. 2024 model + connectivity (FlyWire v630 and v783)
if [ ! -d external/Drosophila_brain_model/.git ]; then
  git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model external/Drosophila_brain_model
else
  echo "skip  external/Drosophila_brain_model"
fi

# 2. FlyWire neuron annotations (Schlegel et al. 2024)
fetch https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
      data/annotations/neuron_annotations.tsv

# 3. FlyWire v783 per-neuron neuropil synapse counts + proofread connections (Zenodo 10676866)
Z=https://zenodo.org/records/10676866/files
for k in pre post; do
  f=per_neuron_neuropil_count_${k}_783.feather
  fetch "$Z/$f?download=1" "data/neuropil/$f"
  # ffsim/lesions.py (E2) reads the same files from data/flywire783/
  mkdir -p data/flywire783
  [ -e "data/flywire783/$f" ] || cp "data/neuropil/$f" "data/flywire783/$f"
done
fetch "$Z/proofread_connections_783.feather?download=1" data/flywire783/proofread_connections_783.feather

# 4. Derived neuron x neuropil table used by ffsim.lesion
PY="${PYTHON:-.venv/bin/python}"
"$PY" -I scripts/build_neuropil_table.py
