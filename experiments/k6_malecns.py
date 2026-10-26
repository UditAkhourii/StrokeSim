"""K6: replicate the CB0246 result in a second connectome (Janelia male CNS v1.0).

Model built by scripts/build_malecns_model.py (165k traced neurons, edges >= 2 synapses,
15.3M edges, Shiu sign rule). Cells mapped by the male CNS flywireType annotation:
CB0246 = GNG033, CB0559 = PRW055, CB0883 = PRW047; sugar SELs CB0262 (GNG540, PN),
CB0212 (GNG550, PN), CB0892 (GNG056, LN), labelled "Yao & Scott 2022: Sugar SEL" by the
male CNS annotators. Sugar input: labellar GRN types LB3b/c/d, LB4b, 100 Hz.

A: sugar both sides, global weight scale 0.8 / 1.0 / 1.2 (x default W_syn), 10 trials.
B: sugar left side only (paper's setup), scale 1.0, 10 trials.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params  # noqa: E402

D = ROOT / 'data' / 'malecns'
DATA = {'path_comp': D / 'Completeness_malecns.csv', 'path_con': D / 'Connectivity_malecns_min2.parquet'}
OUT = ROOT / 'results' / 'k6'
PUMP = ['MN10', 'MNx01', 'CB0700', 'CB0708']
POSITION = ['CB0701', 'CB0762', 'CB0858', 'CB0871', 'CB0911']
SEL_TYPES = ['CB0262', 'CB0212', 'CB0892']


def cells():
    a = pd.read_feather(D / 'ann.feather', columns=['bodyId', 'flywireType', 'type', 'somaSide', 'rootSide', 'status'])
    a = a[a.status == 'Traced']
    a['side'] = a.somaSide.fillna(a.rootSide)
    by = lambda t: a[a.flywireType == t]
    one = lambda t, s: list(by(t).bodyId[by(t).side == s])
    c = {
        'sugar_L': list(a.bodyId[a.type.isin(['LB3b', 'LB3c', 'LB3d', 'LB4b']) & (a.side == 'L')]),
        'sugar_R': list(a.bodyId[a.type.isin(['LB3b', 'LB3c', 'LB3d', 'LB4b']) & (a.side == 'R')]),
        'sel': list(a.bodyId[a.flywireType.isin(SEL_TYPES)]),
        'pump': list(a.bodyId[a.flywireType.isin(PUMP)]),
        'position': list(a.bodyId[a.flywireType.isin(POSITION)]),
    }
    for t in ['CB0246', 'CB0559', 'CB0883']:
        c[f'{t}_L'], c[f'{t}_R'] = one(t, 'L'), one(t, 'R')
    return c


def summarize(r, c):
    m = lambda ids: float(np.mean([r.get(i, 0.0) for i in ids])) if ids else np.nan
    return {'sel6_mean': m(c['sel']), 'pump_mean': m(c['pump']), 'position_mean': m(c['position']),
            'CB0246_L': m(c['CB0246_L']), 'CB0246_R': m(c['CB0246_R']), 'active': int((r > 0).sum())}


def run_set(model, conds, n_run, outdir, t_run_s, c):
    (outdir / 'rates').mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (les, w) in conds.items():
        f = outdir / 'rates' / f'{name}.csv'
        if not f.exists():
            print(f'>>> {name}', flush=True)
            df, _ = model.run(w_scale=w, slnc_ids=les, n_run=n_run)
            rates(df, t_run_s, n_run).to_csv(f)
        rows.append({'cond': name, **summarize(pd.read_csv(f, index_col=0).r, c)})
    d = pd.DataFrame(rows)
    d.to_csv(outdir / 'summary.csv', index=False)
    print(d.round(1).to_string(index=False), flush=True)


def lesions(c):
    return {'intact': [], 'CB0246_L': c['CB0246_L'], 'CB0246_R': c['CB0246_R'],
            'CB0246_both': c['CB0246_L'] + c['CB0246_R'], 'CB0559_both': c['CB0559_L'] + c['CB0559_R'],
            'CB0883_both': c['CB0883_L'] + c['CB0883_R']}


def main():
    c = cells()
    print({k: len(v) for k, v in c.items()}, flush=True)
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    t_run_s = float(p['t_run'] / second)
    n_con = len(pd.read_parquet(DATA['path_con'], columns=['Presynaptic_Index']))

    print('=== A: both sides, weight scales ===', flush=True)
    model = Model(c['sugar_L'] + c['sugar_R'], p, data=DATA)
    conds = {}
    for s in [0.8, 1.0, 1.2]:
        w = None if s == 1.0 else np.full(n_con, s)
        for k, les in lesions(c).items():
            conds[f'x{s}_{k}'] = (les, w)
    run_set(model, conds, 10, OUT / 'both_sides', t_run_s, c)
    del model

    print('=== B: left side only (paper setup) ===', flush=True)
    model = Model(c['sugar_L'], p, data=DATA)
    run_set(model, {k: (les, None) for k, les in lesions(c).items()}, 10, OUT / 'left_only', t_run_s, c)


if __name__ == '__main__':
    main()
