"""Single-process runner around the Shiu et al. LIF model.

Builds the network once, then store()/restore() between trials instead of
rebuilding per trial (Shiu run_exp rebuilds; too slow/memory-heavy on 8 GB).
"""
import sys
from pathlib import Path
from time import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / 'external' / 'Drosophila_brain_model'
sys.path.insert(0, str(SHIU))

from brian2 import Network, mV  # noqa: E402
from model import create_model, poi, default_params  # noqa: E402

V630 = {
    'path_comp': SHIU / '2023_03_23_completeness_630_final.csv',
    'path_con': SHIU / '2023_03_23_connectivity_630_final.parquet',
}


def run(exc_ids, params=None, slnc_ids=(), data=V630, n_run=None, verbose=True):
    """Run n_run trials exciting exc_ids, silencing slnc_ids (FlyWire IDs).

    Returns spike DataFrame with columns t, trial, flywire_id.
    """
    params = dict(default_params if params is None else params)
    n_run = params['n_run'] if n_run is None else n_run

    df_comp = pd.read_csv(data['path_comp'], index_col=0)
    flyid2i = {f: i for i, f in enumerate(df_comp.index)}
    i2flyid = df_comp.index.values

    t0 = time()
    neu, syn, spk_mon = create_model(data['path_comp'], data['path_con'], params)
    pois, neu = poi(neu, [flyid2i[f] for f in exc_ids], [], params)
    if slnc_ids:
        slnc = set(flyid2i[f] for f in slnc_ids)
        mask = pd.Series(syn.i[:]).isin(slnc).values
        w = syn.w[:]
        w[mask] = 0 * mV
        syn.w[:] = w
    net = Network(neu, syn, spk_mon, *pois)
    net.store('init')
    if verbose:
        print(f'    build: {time() - t0:.1f} s')

    rows = []
    for trial in range(n_run):
        t1 = time()
        net.restore('init')
        net.run(params['t_run'])
        i, t = spk_mon.i[:], spk_mon.t[:]
        rows.append(pd.DataFrame({'t': t.astype(float),
                                  'trial': trial, 'flywire_id': i2flyid[i]}))
        if verbose:
            print(f'    trial {trial + 1}/{n_run}: {len(i)} spikes, {time() - t1:.1f} s', flush=True)
    return pd.concat(rows, ignore_index=True)


def rates(df_spk, t_run_s, n_run):
    """Mean and SD firing rate (Hz) per neuron across trials (absent trials = 0)."""
    cnt = df_spk.groupby(['flywire_id', 'trial']).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=range(n_run), fill_value=0) / t_run_s
    return pd.DataFrame({'r': cnt.mean(axis=1), 'std': cnt.std(axis=1, ddof=0)})
