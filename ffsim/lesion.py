"""Lesion sets: neurons to silence for stroke-like perturbations (FlyWire v783)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEUROPIL = ROOT / 'data' / 'neuropil' / 'neuron_neuropil_783.parquet'


def load_neuropil():
    """Long table: flywire_id, neuropil, count, frac (fraction of the neuron's pre+post synapses)."""
    return pd.read_parquet(NEUROPIL)


def neuropil_lesion(neuropil, np_table=None, min_frac=0.5, spare=()):
    """Neurons with >= min_frac of their synapses in `neuropil`, minus `spare` IDs."""
    d = load_neuropil() if np_table is None else np_table
    sel = d[(d.neuropil == neuropil) & (d.frac >= min_frac)].flywire_id
    spare = set(spare)
    return [f for f in sel if f not in spare]


# --- A2: focal (sphere) lesions -------------------------------------------------
# FlyWire annotation positions are in 4 x 4 x 40 nm voxels (checked: pos_x/y/z ranges
# x 22k-226k, y 13k-111k, z 16-6968 voxels -> ~815 x 390 x 280 um incl. optic lobes).
VOXEL_NM = np.array([4.0, 4.0, 40.0])
ANN = ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv'
COMP_783 = ROOT / 'external' / 'Drosophila_brain_model' / 'Completeness_783.csv'
FOCAL_NEUROPILS_EXTRA = ['GNG', 'PRW', 'SAD', 'FLA_L', 'FLA_R']
NON_CENTRAL = ('AME', 'LA_', 'LO_', 'LOP_', 'ME_', 'OCG', 'None')  # optic lobes, ocellar, unassigned


def load_positions(use='pos'):
    """DataFrame indexed by v783 flywire_id with x, y, z in um.

    use='pos'  : annotation pos_x/y/z (a point on the neuron, present for all; for most
                 neurons it lies on the soma or primary neurite), default.
    use='soma' : soma_x/y/z (missing for ~15% of neurons, e.g. sensory afferents; those
                 fall back to pos).
    Only IDs present in Completeness_783 are returned.
    """
    cols = ['root_id', 'pos_x', 'pos_y', 'pos_z', 'soma_x', 'soma_y', 'soma_z']
    a = pd.read_csv(ANN, sep='\t', usecols=cols).drop_duplicates('root_id').set_index('root_id')
    xyz = a[['pos_x', 'pos_y', 'pos_z']].to_numpy(float)
    if use == 'soma':
        s = a[['soma_x', 'soma_y', 'soma_z']].to_numpy(float)
        ok = ~np.isnan(s).any(axis=1)
        xyz[ok] = s[ok]
    p = pd.DataFrame(xyz * VOXEL_NM / 1000, index=a.index, columns=['x', 'y', 'z'])
    ids = pd.read_csv(COMP_783, index_col=0).index
    return p.reindex(ids).dropna()


def sphere_lesion(center_xyz_um, radius_um, pos=None, penumbra_k=None, spare=()):
    """Neurons whose position point (see load_positions) lies within radius_um of center.

    Returns core IDs, or (core_ids, penumbra_ids) if penumbra_k is given, the penumbra
    being the shell radius < d <= radius * penumbra_k. `spare` IDs are excluded from both.
    """
    p = load_positions() if pos is None else pos
    d = np.linalg.norm(p[['x', 'y', 'z']].to_numpy() - np.asarray(center_xyz_um, float), axis=1)
    spare = set(spare)
    core = [f for f in p.index[d <= radius_um] if f not in spare]
    if penumbra_k is None:
        return core
    shell = (d > radius_um) & (d <= radius_um * penumbra_k)
    return core, [f for f in p.index[shell] if f not in spare]


def random_size_matched(n, exclude=(), seed=0, pool=None):
    """n neurons drawn uniformly (no spatial structure) from pool (default all v783 IDs) minus exclude."""
    pool = pd.read_csv(COMP_783, index_col=0).index if pool is None else pd.Index(pool)
    pool = pool[~pool.isin(list(exclude))]
    rng = np.random.default_rng(seed)
    return list(rng.choice(pool.to_numpy(), size=n, replace=False))


def neuropil_centroids(np_table=None, pos=None, neuropils=None, min_frac=0.5):
    """Centroid (um) of each neuropil: neuron positions weighted by synapse fraction there.

    Default neuropil list: central-brain neuropils (no optic lobe / ocellar) incl. GNG,
    PRW, SAD, FLA. Note: neuron position points are mostly somata in the cortex, so
    centroids sit near but not exactly on the neuropil's synaptic mass.
    """
    d = load_neuropil() if np_table is None else np_table
    p = load_positions() if pos is None else pos
    if neuropils is None:
        neuropils = sorted(n for n in d.neuropil.unique() if not n.startswith(NON_CENTRAL))
        neuropils = sorted(set(neuropils) | set(FOCAL_NEUROPILS_EXTRA))
    d = d[d.neuropil.isin(neuropils) & (d.frac >= min_frac)].join(p, on='flywire_id', how='inner')
    w = d.frac.to_numpy()
    out = d[['x', 'y', 'z']].mul(w, axis=0).groupby(d.neuropil).sum().div(d.groupby('neuropil').frac.sum(), axis=0)
    out['n_neurons'] = d.groupby('neuropil').size()
    return out
