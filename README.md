# ffsim — stroke-like lesions in a whole-brain fly model

Code for **"A single relay neuron decides the serotonergic cost of stroke in a whole-brain fly model"**
(Udit Akhouri, Lossfunk).

We take the whole-brain leaky integrate-and-fire (LIF) model of *Drosophila* by
[Shiu et al. (2024)](https://doi.org/10.1038/s41586-024-07763-9), run on the FlyWire connectome
(v783; v630 for replication), and apply stroke-like lesions: silencing all neurons that live in a
neuropil, removing all synapses inside a neuropil ("tissue" lesions), spherical focal lesions with a
penumbra, and size-matched random controls. Sugar gustatory receptor neurons (GRNs) are stimulated
at 100 Hz; readouts are the proboscis-extension motor neuron MN9 (feeding) and serotonergic /
monoaminergic cells of the subesophageal zone (SEZ).

## Key findings

1. **Feeding is robust; the GNG is a single point of failure.** MN9 sugar responses survive neuron
   lesions of 49 of 51 neuropils and almost all tissue lesions. Lesioning the gnathal ganglion (GNG)
   abolishes it.
2. **A PRW stroke selectively removes a serotonergic signal.** Lesioning the prow (PRW, 402 neurons)
   silences three serotonergic SEZ neurons (CB0262 and two CB0892) while MN9 is spared.
   0 of 22 size-matched random lesions reproduce this.
3. **One relay neuron carries it.** Silencing a single neuron, CB0246
   (FlyWire `720575940632602532`), abolishes the serotonergic response (all three cells at 0 Hz,
   MN9 64.6 Hz). The result replicates on the v630 connectome and across W_syn = 0.25–0.35 mV.
4. **Model audit.** PAM dopaminergic neurons are not recruited by taste input. Raising the global
   synaptic weight recruits them only through valence-blind whole-brain ignition (W_syn ≥ 0.49 mV).
   The olfactory subnetwork shows all-or-none ignition.

Readout IDs (v783): MN9 `720575940660219265`; 5-HT cells `720575940636895406`,
`720575940631179831`, `720575940628676457`.

## Quick start

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone <this repo> ffsim && cd ffsim
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
scripts/download_data.sh          # ~ several GB; idempotent
.venv/bin/python experiments/e0_baseline.py --rates 100 --n-run 2   # smoke test
```

Notes on dependencies:

- **`numpy<2.4`** is required. Brian2 2.9.0 calls `ndarray.ptp`, which numpy 2.4 removed.
  `requirements.txt` pins the exact versions used (numpy 2.3.5).
- **Cython** is needed for Brian2's `cython` code-generation target. The first run per connectome
  version compiles the network (about 15 min); later runs reuse the cache.

All scripts are run from the repo root and resolve paths relative to it.

## Data

`scripts/download_data.sh` fetches everything into `external/` and `data/` (both git-ignored):

| What | Source | Destination |
|---|---|---|
| Shiu et al. model + v630/v783 connectivity, `Completeness_783.csv` | `git clone https://github.com/philshiu/Drosophila_brain_model` | `external/Drosophila_brain_model/` |
| FlyWire neuron annotations (Schlegel et al. 2024) | [raw GitHub TSV](https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv) | `data/annotations/neuron_annotations.tsv` |
| Per-neuron neuropil synapse counts, v783 | [Zenodo 10676866](https://zenodo.org/records/10676866) `per_neuron_neuropil_count_{pre,post}_783.feather` | `data/neuropil/`, `data/flywire783/` |
| Proofread connections, v783 | Zenodo 10676866 `proofread_connections_783.feather` | `data/flywire783/` |

`scripts/build_neuropil_table.py` then builds `data/neuropil/neuron_neuropil_783.parquet`: pre and
post counts concatenated, summed per `flywire_id × neuropil`, restricted to the v783 neurons in
`Completeness_783.csv`, with `frac = count / total`. A neuron belongs to a neuropil lesion when
`frac ≥ 0.5`.

## Repository layout

```
ffsim/            library
  sim.py          Shiu LIF runner; Model builds once and store()/restore()s between trials
  lesion.py       neuropil / sphere / random lesion sets, neuropil centroids
  lesions.py      neuropil fractions + lesion sets used by E2
  readouts.py     readout panels from FlyWire annotations (PAM, PPL, MBON, SEZ)
experiments/      one script per experiment (see table)
scripts/          data download + derived-table builder
results_summary/  small CSVs behind the paper's numbers (full outputs go to results/, ignored)
```

## Experiments

| Script | Paper section | Output |
|---|---|---|
| `e0_baseline.py`, `e0_plot.py` | Methods: model validation (Shiu sugar → MN9, v630) | `results/e0/` |
| `e1_reward_readout.py` | Model audit: PAM/PPL/MBON panel, sugar vs bitter | `results/e1/`, `results/e1_v783/` |
| `e1b_readouts.py`, `e1b_analyze.py` | Model audit: SEZ monoaminergic readout, PAM Vm; defines `sez_da_ids.csv` | `results/e1b/` |
| `e1b_odor_probe.py` | Model audit: olfactory all-or-none ignition | `results/e1b_odor/` |
| `n1_wsyn_tradeoff.py`, `n1_plot.py` | Model audit: W_syn sweep, PAM only via ignition (Fig. 3) | `results/n1/` |
| `a1_stroke_atlas.py` | Finding 1/2: single-neuropil neuron-lesion atlas | `results/a1/` |
| `a1b_random_null.py` | Finding 2: 10 random 400-neuron lesions | `results/a1b/` |
| `e2_lesion_atlas.py`, `e2_analyze.py`, `stroke_analysis.py` | Finding 1/2: tissue + neuron lesion atlas, random size controls, confirmation | `results/e2/`, `results/e2_confirm/` |
| `a2_sphere_lesions.py` | Finding 2: spherical focal strokes with penumbra | `results/a2/` |
| `a3_cb0246_bottleneck.py` | Finding 3: CB0246 single-neuron silencing | `results/a3/` |
| `a5r_replication.py` | Finding 3: v630 + W_syn replication; round-stroke null | `results/replication/` |
| `a4_pruning.py` | Exploratory (synaptic pruning), not in the paper | `results/a4/` |

Exact commands, in order, are in [REPRODUCE.md](REPRODUCE.md). `results_summary/` holds the small
CSVs that back the numbers above, including `a1b_random400_null.csv` (MN9 and 5-HT rates per random
seed).

## Runtime and memory

- About 3 GB RAM per simulation process. On an 8 GB machine run one or two at a time.
- About 5 s per 1 s simulated trial after compilation (Apple-silicon laptop).
- First compile about 15 min per connectome version (v630, v783).

## Limitations

- The Shiu model is a point-neuron LIF network with connectome weights and uniform parameters. It has no
  neuromodulation, plasticity, gap junctions or glia, and neurotransmitter signs are predicted.
- "Stroke" means silencing neurons (outgoing weights set to 0) or deleting synapses. There is no
  ischaemic cascade, no recovery and no rewiring.
- "Serotonergic" is an annotation-level label. The model does not simulate 5-HT release.
- Rates are averaged over 10–30 Poisson trials. Effects near threshold depend on W_syn, so we report a
  W_syn sweep.
- Results are for sugar GRN input only.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite Shiu et al. 2024 and the FlyWire papers below.

## License

MIT, see [LICENSE](LICENSE). External data keep their own licenses.

## Acknowledgements

- Shiu, P. K. et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature (2024). Model code: https://github.com/philshiu/Drosophila_brain_model
- Dorkenwald, S. et al. *Neuronal wiring diagram of an adult brain.* Nature (2024), and the FlyWire Consortium.
- Schlegel, P. et al. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature (2024). Annotations: https://github.com/flyconnectome/flywire_annotations
