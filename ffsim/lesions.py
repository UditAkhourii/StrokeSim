"""Lesion sets. Neuropil assignment from FlyWire v783 per-neuron neuropil synapse counts (Zenodo 10676866)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NP_DIR = ROOT / 'data' / 'flywire783'


def neuropil_fractions(ids):
    """Fraction of each neuron's synapses (pre + post) in each neuropil. Rows: root IDs in `ids`."""
    parts = []
    for k in ['pre', 'post']:
        d = pd.read_feather(NP_DIR / f'per_neuron_neuropil_count_{k}_783.feather')
        d = d.rename(columns={f'{k}_pt_root_id': 'flywire_id'})
        parts.append(d[d.flywire_id.isin(ids)])
    d = pd.concat(parts).groupby(['flywire_id', 'neuropil'])['count'].sum().unstack(fill_value=0)
    return d.div(d.sum(axis=1), axis=0)


def neuropil_lesions(frac, thresh=0.5, protect=()):
    """{neuropil: [root IDs with >= thresh of synapses there]}, excluding protected IDs."""
    protect = set(protect)
    out = {}
    for np_ in frac.columns:
        if np_ == 'UNASGD':
            continue
        sel = frac.index[frac[np_] >= thresh]
        out[np_] = [f for f in sel if f not in protect]
    return out


def random_lesion(ids, n, seed, protect=()):
    pool = np.array(sorted(set(ids) - set(protect)))
    return list(np.random.default_rng(seed).choice(pool, size=n, replace=False))
