"""D1: can the serotonin signal be separated from swallowing?

Pre-registered 2026-10-27 (see "Experiment D1 - Serotonin vs Swallowing Dissociation.md").

Q1 showed that silencing CB0246 L removes the sugar-evoked 5-HT response but also knocks
down pharyngeal pump motor neurons. D1 asks whether the two can be dissociated by cutting
edge sets instead of whole neurons, using Model.run(w_scale=...).

Conditions
  C0 intact
  C1 CB0246 L silenced (full knockout; reproduces Q1)
  C2 cut CB0246 L -> SEL6 synapses only
  C3 cut every CB0246 L output EXCEPT onto SEL6
  C4 C2 for both CB0246 L and R
  C5 silence all 6 SEL 5-HT cells
  C6 silence the 5 non-5-HT Q1-module neurons with the most synapses onto pump MNs

v783, default W_syn, 20 left sugar GRNs at 100 Hz, 1 s, 20 trials per condition.

usage: python experiments/d1_dissociation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second, prefs  # noqa: E402

prefs.codegen.runtime.cython.cache_dir = str(ROOT / '.cython_cache')

from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402
from experiments.k2_bilateral import SEL, CB0246_L, CB0246_R  # noqa: E402

OUT = ROOT / 'results' / 'd1'
S = ROOT / 'results_summary'
PUMP_TYPES = ['MN10', 'MNx01', 'CB0700', 'CB0708']
PAPER3 = [720575940636895406, 720575940631179831, 720575940628676457]
SEL6 = sorted(SEL.values())
N_RUN = 20


def pump_ids(ids):
    ann = pd.read_csv(ROOT / 'data/annotations/neuron_annotations.tsv', sep='\t',
                      usecols=['root_id', 'cell_type'], low_memory=False).drop_duplicates('root_id')
    return sorted(f for f in ann.root_id[ann.cell_type.isin(PUMP_TYPES)] if f in ids)


def module_ids():
    q1 = pd.read_csv(S / 'q1_brainwide_changed.csv')
    return list(q1.flywire_id[q1.knockout == 'CB0246'])


def edge_rows(pump):
    """Row indices into Connectivity_783.parquet for the three cut sets, plus the C6 set."""
    con = pd.read_parquet(V783['path_con'], columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'])
    pre_L = con.Presynaptic_ID.values == CB0246_L
    pre_LR = np.isin(con.Presynaptic_ID.values, [CB0246_L, CB0246_R])
    post_sel = np.isin(con.Postsynaptic_ID.values, SEL6)
    cuts = {
        'C2': np.flatnonzero(pre_L & post_sel),
        'C3': np.flatnonzero(pre_L & ~post_sel),
        'C4': np.flatnonzero(pre_LR & post_sel),
    }
    # C6: non-5HT Q1-module neurons ranked by synapse count onto pump MNs
    mod = [f for f in module_ids() if f not in SEL6]
    sub = con[con.Presynaptic_ID.isin(mod) & con.Postsynaptic_ID.isin(pump)]
    rank = sub.groupby('Presynaptic_ID').Connectivity.sum().sort_values(ascending=False)
    c6 = list(rank.index[:5])
    n_con = len(con)
    del con, sub
    return cuts, c6, rank, n_con


def group_trials(df_spk, t_run_s, pump):
    """Per-trial group readouts (Hz): SEL6 mean, paper-3 mean, pump sum, MN9."""
    cnt = df_spk.groupby(['flywire_id', 'trial']).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=range(N_RUN), fill_value=0) / t_run_s
    def take(ids, how):
        d = cnt.reindex(ids).fillna(0.0)
        return d.mean(axis=0) if how == 'mean' else d.sum(axis=0)
    return pd.DataFrame({'trial': range(N_RUN),
                         'sel6_mean': take(SEL6, 'mean').values,
                         'paper3_mean': take(PAPER3, 'mean').values,
                         'pump_sum': take(pump, 'sum').values,
                         'mn9': take([ID_MN9], 'sum').values})


def changed(base, cond, excl):
    """q1_brainwide rule: |delta| >= 50% of larger rate, >= 2 Hz, |z| >= 3."""
    d = base.add_suffix('_int').join(cond.add_suffix('_ko'), how='outer').fillna(0.0)
    d = d.drop(index=[i for i in excl if i in d.index])
    d['delta'] = d.r_ko - d.r_int
    se = np.sqrt(d.std_int ** 2 / N_RUN + d.std_ko ** 2 / N_RUN).clip(lower=0.1)
    d['z'] = d.delta / se
    big = d[['r_int', 'r_ko']].max(axis=1)
    d['changed'] = (d.delta.abs() >= 0.5 * big) & (d.delta.abs() >= 2) & (d.z.abs() >= 3)
    return d


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    (OUT / 'trials').mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    pump = pump_ids(ids)
    assert len(sugar) == 20 and len(pump) == 10, (len(sugar), len(pump))
    cuts, c6, rank, n_con = edge_rows(pump)
    print(f'cut sets (rows): ' + ', '.join(f'{k}={len(v)}' for k, v in cuts.items()), flush=True)
    print('C6 set (pre-registered rule: top 5 non-5HT module neurons by synapses onto pump MNs):',
          flush=True)
    print(rank.head(5).to_string(), flush=True)

    p = dict(default_params)
    p.update(r_poi=100 * Hz, n_run=N_RUN)
    t_run_s = float(p['t_run'] / second)

    conds = {
        'C0_intact':      dict(),
        'C1_CB0246L_ko':  dict(slnc_ids=[CB0246_L]),
        'C2_cut_L_to_SEL': dict(cut='C2'),
        'C3_cut_L_except_SEL': dict(cut='C3'),
        'C4_cut_LR_to_SEL': dict(cut='C4'),
        'C5_SEL6_ko':     dict(slnc_ids=SEL6),
        'C6_module5_ko':  dict(slnc_ids=c6),
    }
    model = None
    for name, spec in conds.items():
        f = OUT / 'rates' / f'{name}.csv'
        if f.exists():
            continue
        if model is None:
            model = Model(sugar, p, data=V783)
        w_scale = None
        if 'cut' in spec:
            w_scale = np.ones(n_con)
            w_scale[cuts[spec['cut']]] = 0.0
        print(f'>>> {name}', flush=True)
        df, _ = model.run(w_scale=w_scale, slnc_ids=spec.get('slnc_ids', ()), n_run=N_RUN)
        rates(df, t_run_s, N_RUN).to_csv(f)
        group_trials(df, t_run_s, pump).to_csv(OUT / 'trials' / f'{name}.csv', index=False)
        del df, w_scale
    del model

    # ---- summary ----
    base = pd.read_csv(OUT / 'rates' / 'C0_intact.csv', index_col=0)
    mod48 = module_ids()
    rows = []
    for name, spec in conds.items():
        r = pd.read_csv(OUT / 'rates' / f'{name}.csv', index_col=0)
        tr = pd.read_csv(OUT / 'trials' / f'{name}.csv')
        excl = list(spec.get('slnc_ids', ())) + sugar
        d = changed(base, r, excl)
        ch = d[d.changed]
        row = {'cond': name,
               'active': int((d.r_ko > 0).sum()),
               'n_changed_brainwide': int(len(ch)),
               'n_changed_module48': int(d.loc[[i for i in mod48 if i in d.index]].changed.sum()),
               'module48_tested': len([i for i in mod48 if i in d.index])}
        for k in ['sel6_mean', 'paper3_mean', 'pump_sum', 'mn9']:
            row[f'{k}_Hz'] = tr[k].mean()
            row[f'{k}_sd'] = tr[k].std(ddof=0)
        rows.append(row)
    res = pd.DataFrame(rows).set_index('cond')
    c0 = res.loc['C0_intact']
    for k in ['sel6_mean', 'paper3_mean', 'pump_sum', 'mn9']:
        res[f'{k}_rel'] = res[f'{k}_Hz'] / c0[f'{k}_Hz']
        # SD of the ratio, intact SD propagated
        res[f'{k}_rel_sd'] = res[f'{k}_sd'] / c0[f'{k}_Hz']
    res['active_rel'] = res.active / c0.active
    res.to_csv(OUT / 'summary.csv')
    res.to_csv(S / 'd1_summary.csv')
    cols = ['sel6_mean_rel', 'paper3_mean_rel', 'pump_sum_rel', 'mn9_rel',
            'n_changed_module48', 'n_changed_brainwide', 'active_rel']
    pd.set_option('display.width', 200)
    print('\n=== D1 summary (normalised to C0) ===')
    print(res[cols].round(3).to_string())
    print('\nabsolute Hz:')
    print(res[[c for c in res.columns if c.endswith('_Hz') or c.endswith('_sd')]].round(2).to_string())


if __name__ == '__main__':
    main()
