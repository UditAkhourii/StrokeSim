"""Single-process runner around the Shiu et al. LIF model.

Builds the network once, then store()/restore() between trials instead of
rebuilding per trial (Shiu run_exp rebuilds; too slow/memory-heavy on 8 GB).
"""
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / 'external' / 'Drosophila_brain_model'
sys.path.insert(0, str(SHIU))

from brian2 import Network, PoissonGroup, StateMonitor, Synapses, mV, ms  # noqa: E402
from model import create_model, default_params  # noqa: E402

V630 = {
    'path_comp': SHIU / '2023_03_23_completeness_630_final.csv',
    'path_con': SHIU / '2023_03_23_connectivity_630_final.parquet',
}
V783 = {
    'path_comp': SHIU / 'Completeness_783.csv',
    'path_con': SHIU / 'Connectivity_783.parquet',
}


def poisson_drive(neu, exc, params):
    """Poisson input to neurons exc, equivalent to Shiu's per-neuron PoissonInput(N=1).

    One PoissonGroup + one Synapses object: rate is a runtime variable, so changing
    the rate or the input set does not trigger Cython recompilation (PoissonInput
    bakes the rate into generated code and compiles one module per target neuron).
    """
    pg = PoissonGroup(len(exc), rates=params['r_poi'], name='poisson_drive')
    syn = Synapses(pg, neu, 'w_poi : volt', on_pre='v += w_poi', name='poisson_syn')
    syn.connect(i=list(range(len(exc))), j=list(exc))
    syn.w_poi = params['w_syn'] * params['f_poi']
    neu.rfc[list(exc)] = 0 * ms  # no refractory period for Poisson targets (as Shiu)
    return [pg, syn]


def run(exc_ids, params=None, slnc_ids=(), data=V630, n_run=None, verbose=True,
        record_v=(), w_scale=None, scale_ids=(), scale_factor=1.0):
    """Run n_run trials exciting exc_ids, silencing slnc_ids (FlyWire IDs).

    Returns spike DataFrame with columns t, trial, flywire_id.
    If record_v (FlyWire IDs) is given, also returns a DataFrame of mean
    depolarisation above rest (mV, 1 ms sampling) per neuron and trial.
    w_scale: optional array aligned to the rows of data['path_con']; multiplies
    each connection's weight (tissue lesions).
    scale_ids / scale_factor: outgoing weights of scale_ids are multiplied by
    scale_factor (partial loss, e.g. stroke penumbra); applied like silencing.
    """
    params = dict(default_params if params is None else params)
    n_run = params['n_run'] if n_run is None else n_run

    df_comp = pd.read_csv(data['path_comp'], index_col=0)
    flyid2i = {f: i for i, f in enumerate(df_comp.index)}
    i2flyid = df_comp.index.values

    t0 = time()
    neu, syn, spk_mon = create_model(data['path_comp'], data['path_con'], params)
    pois = poisson_drive(neu, [flyid2i[f] for f in exc_ids], params)
    if w_scale is not None:
        # connect() keeps the row order of the connectivity table; check before scaling
        con = pd.read_parquet(data['path_con'], columns=['Presynaptic_Index'])
        assert len(w_scale) == len(con) and (syn.i[:] == con.Presynaptic_Index.values).all()
        syn.w[:] = syn.w[:] * w_scale
    if slnc_ids:
        slnc = set(flyid2i[f] for f in slnc_ids)
        mask = pd.Series(syn.i[:]).isin(slnc).values
        w = syn.w[:]
        w[mask] = 0 * mV
        syn.w[:] = w
    if scale_ids and scale_factor != 1.0:
        scl = set(flyid2i[f] for f in scale_ids)
        mask = pd.Series(syn.i[:]).isin(scl).values
        w = syn.w[:]
        w[mask] = w[mask] * scale_factor
        syn.w[:] = w
    objs = [neu, syn, spk_mon, *pois]
    if record_v:
        v_ids = [f for f in record_v if f in flyid2i]
        v_mon = StateMonitor(neu, 'v', record=[flyid2i[f] for f in v_ids], dt=1 * ms)
        objs.append(v_mon)
    net = Network(*objs)
    net.store('init')
    if verbose:
        print(f'    build: {time() - t0:.1f} s')

    rows, v_rows = [], []
    for trial in range(n_run):
        t1 = time()
        net.restore('init')
        net.run(params['t_run'])
        i, t = spk_mon.i[:], spk_mon.t[:]
        rows.append(pd.DataFrame({'t': t.astype(float),
                                  'trial': trial, 'flywire_id': i2flyid[i]}))
        if record_v:
            dv = (v_mon.v[:] - params['v_0']).mean(axis=1) / mV
            v_rows.append(pd.DataFrame({'flywire_id': v_ids, 'trial': trial, 'dv_mV': dv}))
        if verbose:
            print(f'    trial {trial + 1}/{n_run}: {len(i)} spikes, {time() - t1:.1f} s', flush=True)
    df_spk = pd.concat(rows, ignore_index=True)
    if record_v:
        return df_spk, pd.concat(v_rows, ignore_index=True)
    return df_spk


def rates(df_spk, t_run_s, n_run):
    """Mean and SD firing rate (Hz) per neuron across trials (absent trials = 0)."""
    cnt = df_spk.groupby(['flywire_id', 'trial']).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=range(n_run), fill_value=0) / t_run_s
    return pd.DataFrame({'r': cnt.mean(axis=1), 'std': cnt.std(axis=1, ddof=0)})


class Model:
    """Network built once; lesion conditions applied by rewriting synaptic weights.

    For sweeps over many lesions: avoids rebuilding the 15M-synapse network per condition.
    Stimulus (exc_ids, rate) and recorded Vm set are fixed at construction.
    """

    def __init__(self, exc_ids, params=None, data=V630, record_v=()):
        self.params = dict(default_params if params is None else params)
        df_comp = pd.read_csv(data['path_comp'], index_col=0)
        self.flyid2i = {f: i for i, f in enumerate(df_comp.index)}
        self.i2flyid = df_comp.index.values
        neu, syn, self.spk_mon = create_model(data['path_comp'], data['path_con'], self.params)
        pois = poisson_drive(neu, [self.flyid2i[f] for f in exc_ids], self.params)
        objs = [neu, syn, self.spk_mon, *pois]
        self.v_ids = [f for f in record_v if f in self.flyid2i]
        self.v_mon = None
        if self.v_ids:
            self.v_mon = StateMonitor(neu, 'v', record=[self.flyid2i[f] for f in self.v_ids], dt=1 * ms)
            objs.append(self.v_mon)
        self.syn = syn
        self.pre = syn.i[:].copy()
        self.w0 = syn.w[:].copy()  # Quantity, volts
        self.net = Network(*objs)
        self.net.store('pristine')  # empty monitors, t=0, intact weights

    def run(self, w_scale=None, slnc_ids=(), n_run=None, scale_ids=(), scale_factor=1.0):
        """Returns (spike DataFrame, Vm DataFrame or None) for the given lesion."""
        n_run = self.params['n_run'] if n_run is None else n_run
        # local names must not collide with model variables (brian2 resolves the caller namespace)
        w_new = self.w0 if w_scale is None else self.w0 * w_scale
        if slnc_ids:
            mask = np.isin(self.pre, [self.flyid2i[f] for f in slnc_ids])
            w_new = w_new.copy()
            w_new[mask] = 0 * mV
        if scale_ids and scale_factor != 1.0:
            mask = np.isin(self.pre, [self.flyid2i[f] for f in scale_ids])
            w_new = w_new.copy()
            w_new[mask] = w_new[mask] * scale_factor
        self.net.restore('pristine')
        self.syn.w[:] = w_new
        self.net.store('init')
        rows, v_rows = [], []
        for trial in range(n_run):
            self.net.restore('init')
            self.net.run(self.params['t_run'])
            i, t = self.spk_mon.i[:], self.spk_mon.t[:]
            rows.append(pd.DataFrame({'t': t.astype(float), 'trial': trial, 'flywire_id': self.i2flyid[i]}))
            if self.v_mon is not None:
                dv = (self.v_mon.v[:] - self.params['v_0']).mean(axis=1) / mV
                v_rows.append(pd.DataFrame({'flywire_id': self.v_ids, 'trial': trial, 'dv_mV': dv}))
        df_v = pd.concat(v_rows, ignore_index=True) if v_rows else None
        return pd.concat(rows, ignore_index=True), df_v
