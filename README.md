# ffsim

Whole-brain *Drosophila* LIF sim (FlyWire connectome) under stroke-like and neurodevelopmental perturbations.
Research plan: Obsidian note `Lossfunk/Fruitfly Stroke Depression Research/FFSIM - ...md`.

## Setup

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model external/Drosophila_brain_model
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python brian2 "numpy<2.4" pandas pyarrow joblib matplotlib cython
```

`numpy<2.4` is required: brian2 2.9.0 uses `ndarray.ptp`, removed in numpy 2.4.

## Experiments

- `experiments/e0_baseline.py`: reproduce Shiu et al. 2024 sugar GRN -> MN9 (FlyWire v630).
  `.venv/bin/python experiments/e0_baseline.py --rates 100 150 200 --n-run 30 --n-proc 2`

Memory: ~3 GB peak per worker process; keep `--n-proc` at 2 on an 8 GB machine.
