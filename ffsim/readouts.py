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


# Odour stand-ins: ORN glomeruli activated by each stimulus
ODORS = {
    'vinegar': ['DM1', 'DM4', 'VM2'],               # Semmelhack & Wang 2009 (attractive)
    'geosmin': ['DA2'],                             # Stensmyr et al. 2012 (aversive)
    'co2': ['V'],                                   # Suh et al. 2004 (aversive)
    'broad': ['DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6', 'DL1', 'DL5', 'VA2', 'VA6', 'VM2'],
}


def orn_ids(glomeruli, ann=None, ids=None):
    """Root IDs of ORNs (both antennae) innervating the given glomeruli."""
    a = load_annotations() if ann is None else ann
    o = a[(a.cell_class == 'olfactory') & a.cell_type.isin([f'ORN_{g}' for g in glomeruli])]
    out = o.root_id.tolist()
    return [f for f in out if f in ids] if ids is not None else out


def olfactory_panel(ann=None, ids=None):
    """ALPNs, Kenyon cells and APL with group labels."""
    a = load_annotations() if ann is None else ann
    d = a[a.cell_class.isin(['ALPN', 'Kenyon_Cell']) | (a.cell_type == 'APL')].copy()
    d['grp'] = d.cell_class.map({'ALPN': 'PN', 'Kenyon_Cell': 'KC'}).fillna('APL')
    d['sub'] = d.cell_sub_class
    d = d[['root_id', 'grp', 'sub', 'cell_type', 'side']].rename(columns={'root_id': 'flywire_id'})
    if ids is not None:
        d = d[d.flywire_id.isin(ids)]
    return d.reset_index(drop=True)


def sez_da_panel(ann=None, ids=None, path=ROOT / 'results' / 'e1b' / 'sez_da_ids.csv'):
    """Sugar-specific dopaminergic (predicted) non-MB neurons, defined in E1b."""
    a = load_annotations() if ann is None else ann
    sel = pd.read_csv(path).flywire_id
    d = a[a.root_id.isin(sel)][['root_id', 'cell_type', 'side', 'top_nt', 'top_nt_conf']]
    d = d.rename(columns={'root_id': 'flywire_id'}).assign(grp='SEZ_DA')
    if ids is not None:
        d = d[d.flywire_id.isin(ids)]
    return d.reset_index(drop=True)
