"""Q1: what else changes across the brain when CB0246 is silenced?

No simulation. Compares per-neuron rates, intact (results/a1/rates/INTACT.csv, 5 trials) vs
single-neuron knockouts (results/a3/rates/*.csv, 10 trials): CB0246, plus CB0208 and the
CB1366 pair as knockouts of strong PRW inputs that do not affect the 5-HT cells.
v783, default W_syn, sugar GRNs at 100 Hz.

A neuron counts as changed when |delta| >= 50% of the larger rate, |delta| >= 2 Hz, and
|z| >= 3 (z from trial SDs).

usage: python experiments/q1_brainwide.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e0_baseline import NEU_SUGAR  # noqa: E402

R = ROOT / 'results'
S = ROOT / 'results_summary'
HT = [720575940636895406, 720575940631179831, 720575940628676457]
CB0246 = 720575940632602532
N_INTACT, N_KO = 5, 10


def load(p):
    return pd.read_csv(p, index_col=0)


def compare(base, ko, silenced):
    d = base.add_suffix('_int').join(ko.add_suffix('_ko'), how='outer').fillna(0.0)
    d = d.drop(index=[i for i in silenced if i in d.index])
    d = d.drop(index=[i for i in NEU_SUGAR if i in d.index])  # stimulated inputs
    d['delta'] = d.r_ko - d.r_int
    se = np.sqrt(d.std_int ** 2 / N_INTACT + d.std_ko ** 2 / N_KO).clip(lower=0.1)
    d['z'] = d.delta / se
    big = d[['r_int', 'r_ko']].max(axis=1)
    d['changed'] = (d.delta.abs() >= 0.5 * big) & (d.delta.abs() >= 2) & (d.z.abs() >= 3)
    return d


def main():
    ann = pd.read_csv(ROOT / 'data/annotations/neuron_annotations.tsv', sep='\t', low_memory=False,
                      usecols=['root_id', 'super_class', 'cell_class', 'cell_type', 'top_nt'])
    ann = ann.set_index('root_id')
    base = load(R / 'a1/rates/INTACT.csv')
    kos = {'CB0246': [CB0246], 'CB0208': None, 'CB1366_x2': None}
    rows, detail = [], []
    for name in kos:
        d = compare(base, load(R / f'a3/rates/{name}.csv'), kos[name] or [])
        if kos[name] is None:  # silenced ids are the ones that drop to exactly 0 from >0 and are named
            d = d[~((d.r_ko == 0) & ann.reindex(d.index).cell_type.eq(name.split('_')[0]).to_numpy())]
        d = d.join(ann, how='left')
        ch = d[d.changed]
        rows.append({'knockout': name, 'active_intact': int((d.r_int > 0).sum()),
                     'active_ko': int((d.r_ko > 0).sum()), 'n_changed': len(ch),
                     'n_down': int((ch.delta < 0).sum()), 'n_up': int((ch.delta > 0).sum()),
                     'frac_changed': len(ch) / max((d[['r_int', 'r_ko']].max(axis=1) > 0).sum(), 1),
                     'sum_abs_delta_Hz': d.delta.abs().sum(),
                     'by_super_class': ch.super_class.value_counts().to_dict()})
        detail.append(ch.assign(knockout=name).reset_index())
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 220, 'display.max_colwidth', 120)
    print(res.to_string(index=False))
    det = pd.concat(detail)
    cols = ['knockout', 'flywire_id', 'cell_type', 'super_class', 'cell_class', 'top_nt', 'r_int', 'r_ko', 'delta', 'z']
    det = det.rename(columns={'index': 'flywire_id'})[cols].sort_values(['knockout', 'delta'])
    det['is_5HT'] = det.flywire_id.isin(HT)
    print('\nchanged neurons, CB0246 knockout:\n', det[det.knockout == 'CB0246'].to_string(index=False))
    res.to_csv(S / 'q1_brainwide_summary.csv', index=False)
    det.to_csv(S / 'q1_brainwide_changed.csv', index=False)


if __name__ == '__main__':
    main()
