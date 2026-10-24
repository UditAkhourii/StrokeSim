"""Readout neuron sets from FlyWire annotations (Schlegel et al. 2024; v783 root IDs)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv'


def load_annotations():
    return pd.read_csv(ANN, sep='\t', low_memory=False)


def reward_panel(ann=None, ids=None):
    """DANs (PAM, PPL1, PPL2) and MBONs with group labels.

    ids: optional set of root IDs present in the connectome version in use;
    neurons whose v783 ID is absent there are dropped (proofread since v630).
    """
    a = load_annotations() if ann is None else ann
    d = a[a.cell_class.isin(['DAN', 'MBON'])].copy()
    d['grp'] = d.cell_type.astype(str).str.extract(r'^(PAM|PPL1|PPL2|MBON)')[0]
    d = d[['root_id', 'grp', 'cell_type', 'side', 'top_nt']].rename(columns={'root_id': 'flywire_id'})
    if ids is not None:
        d = d[d.flywire_id.isin(ids)]
    return d.reset_index(drop=True)
