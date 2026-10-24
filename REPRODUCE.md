# Reproducing the results

Run everything from the repo root. `PY=.venv/bin/python`. Runtimes assume about 5 s per trial after
compilation and one process at a time. The first run per connectome version also needs about 15 min
of Cython compilation. Plan for about 3 GB RAM per process.

Several scripts skip conditions whose output CSV already exists, so an interrupted run can be resumed.
Order matters: `results/e1b/sez_da_ids.csv` (step E1b) is read by A1, A1b, A2, A4, N1 and E2.

## 0. Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
scripts/download_data.sh                    # clone Shiu repo, annotations, Zenodo tables, build neuropil parquet
PY=.venv/bin/python
```

## E0. Baseline: sugar GRN → MN9 (v630), about 15 min compile + 10 min

```bash
$PY experiments/e0_baseline.py --rates 100 150 200 --n-run 30
$PY experiments/e0_plot.py
```

## E1. Reward panel, sugar vs bitter, about 10 min per version

```bash
$PY experiments/e1_reward_readout.py --version 630
$PY experiments/e1_reward_readout.py --version 783     # + about 15 min first v783 compile
```

## E1b. SEZ readout, PAM Vm, olfactory ignition, about 20 min

```bash
$PY experiments/e1b_readouts.py --n-run 30
$PY experiments/e1b_analyze.py                         # writes results/e1b/summary_*.csv, sez_da_ids.csv
# olfactory probe: one 200 ms trial per call (about 10 s each); odor_sweep.txt is these lines collected
for r in 60 70 80 90 100; do $PY experiments/e1b_odor_probe.py orn DA2 $r; done
```

## N1. W_syn sweep (Fig. 3), 10 W_syn × 2 stimuli × 10 trials, about 20 min

```bash
$PY experiments/n1_wsyn_tradeoff.py
$PY experiments/n1_plot.py
```

## A1. Single-neuropil neuron-lesion atlas, about 51 neuropils × 5 trials, about 25 min

```bash
$PY experiments/a1_stroke_atlas.py --n-run 5
```

## A1b. Random 400-neuron null, 10 seeds × 10 trials, about 10 min

```bash
$PY experiments/a1b_random_null.py
```

## A2. Spherical focal strokes (PRW and others), variable runtime

```bash
$PY experiments/a2_sphere_lesions.py --dry-run         # prints plan and lesion sizes
$PY experiments/a2_sphere_lesions.py --n-run 10 --only PRW
```

## A3. CB0246 single-neuron bottleneck, 4 conditions × 10 trials, about 5 min

```bash
$PY experiments/a3_cb0246_bottleneck.py
```

## E2. Tissue / neuron lesion atlas + confirmation

E2 also reads `results/e2/neuropil_fractions.parquet` and `results/e2/conn_neuropil_frac.parquet`.
The builder for these intermediate files is **not in this repository yet** (see "Known gaps").

```bash
# screen: 5 trials, tissue lesions only, optic lobe skipped (about 68 conditions, about 30 min)
$PY experiments/e2_lesion_atlas.py --n-run 5 --order tissue --skip-visual
# full atlas: 146 conditions × 15 trials, about 3 h
$PY experiments/e2_lesion_atlas.py --n-run 15
# confirmation of hits: 20 trials, about 20 min
$PY experiments/e2_lesion_atlas.py --n-run 20 --n-intact 20 --outdir e2_confirm \
    --only intact tissue:GNG tissue:IB_L tissue:IB_R tissue:MB_PED_L tissue:PRW tissue:SCL_R \
           random:1000 neuron:GNG neuron:PRW
$PY experiments/e2_analyze.py
$PY experiments/stroke_analysis.py
```

## Replication: v630 + W_syn 0.25–0.35 mV; round-stroke null, about 1 h

```bash
$PY experiments/a5r_replication.py                     # also compiles v630 if not cached
```

## Paper numbers → files

The `results_summary/` copies of: `e0/summary.csv`, `n1/summary.csv`, `a1/atlas.csv`,
`e2/tissue_screen.csv`, `e2_confirm/summary.csv`, `a3/summary.csv`, `replication/*.csv`,
`e1b/summary_*.csv`, `e1b/sez_da_ids.csv`, `e1b_odor/odor_sweep.txt`.
`a1b_random400_null.csv` is MN9 and 5-HT rates extracted from `results/a1b/rates/random400_s*.csv`.

## Known gaps

- The builder for `results/e2/neuropil_fractions.parquet` and `conn_neuropil_frac.parquet` is missing
  from the repository. They derive from `data/flywire783/proofread_connections_783.feather` and the
  per-neuron neuropil tables.
- The exact E2 screen/confirm flag sets are reconstructed from logs (`results/e2_*.log`). The condition
  lists match, but trial counts should be checked against `results/e2*/conds`.
