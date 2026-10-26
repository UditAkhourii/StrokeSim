"""K4: robustness and mirror tests for the CB0246 result.

A, robustness: sugar on both sides (K3 input). For 10 seeds, every connection weight is
multiplied by lognormal noise (sigma 0.3, mean 1); run intact and CB0246 L silenced.
B, mirror: sugar on the RIGHT side only (K3's 21 type-matched right GRNs); intact,
CB0246 L, CB0246 R. Does the critical side follow the input side?
Readouts: 6 serotonin cells, pharyngeal pump MNs (MN10, MNx01, CB0700, CB0708),
proboscis positioning MNs (CB0701 = MN9, CB0762, CB0858, CB0871, CB0911).
v783, default W_syn, 100 Hz, 10 trials (A) / 20 trials (B).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR  # noqa: E402
from experiments.k2_bilateral import SEL, CB0246_L, CB0246_R  # noqa: E402
from experiments.k3_bilateral_sugar import right_sugar  # noqa: E402

OUT = ROOT / 'results' / 'k4'
PUMP = ['MN10', 'MNx01', 'CB0700', 'CB0708']
POSITION = ['CB0701', 'CB0762', 'CB0858', 'CB0871', 'CB0911']


def groups():
    a = pd.read_csv(ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv', sep='\t',
                    usecols=['root_id', 'cell_type'], low_memory=False).drop_duplicates('root_id')
    return (list(a.root_id[a.cell_type.isin(PUMP)]), list(a.root_id[a.cell_type.isin(POSITION)]))


def summarize(r, pump, pos):
    return {'sel6_mean': np.mean([r.get(v, 0.0) for v in SEL.values()]),
            'pump_mean': np.mean([r.get(v, 0.0) for v in pump]),
            'position_mean': np.mean([r.get(v, 0.0) for v in pos]),
            'CB0246_L': r.get(CB0246_L, 0.0), 'CB0246_R': r.get(CB0246_R, 0.0)}


def run_set(model, conds, n_run, outdir, t_run_s, pump, pos):
    (outdir / 'rates').mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (les, w_scale) in conds.items():
        f = outdir / 'rates' / f'{name}.csv'
        if not f.exists():
            print(f'>>> {name}', flush=True)
            df, _ = model.run(w_scale=w_scale, slnc_ids=les, n_run=n_run)
            rates(df, t_run_s, n_run).to_csv(f)
        rows.append({'cond': name, **summarize(pd.read_csv(f, index_col=0).r, pump, pos)})
    d = pd.DataFrame(rows)
    d.to_csv(outdir / 'summary.csv', index=False)
    print(d.round(1).to_string(index=False), flush=True)
    return d


def main():
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    left = [f for f in NEU_SUGAR if f in ids]
    right = right_sugar(ids)
    pump, pos = groups()
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    t_run_s = float(p['t_run'] / second)
    n_con = len(pd.read_parquet(V783['path_con'], columns=['Presynaptic_Index']))

    print('=== A: robustness (both sides, weight noise) ===', flush=True)
    model = Model(left + right, p, data=V783)
    conds = {}
    for s in range(10):
        w = np.random.default_rng(1000 + s).lognormal(-0.3 ** 2 / 2, 0.3, n_con)
        conds[f's{s}_intact'] = ([], w)
        conds[f's{s}_CB0246_L'] = ([CB0246_L], w)
    a = run_set(model, conds, 10, OUT / 'robust', t_run_s, pump, pos)
    a['seed'] = a.cond.str.split('_').str[0]
    piv = a.pivot_table(index='seed', columns=a.cond.str.contains('CB0246'), values=['sel6_mean', 'pump_mean', 'position_mean'])
    rel = pd.DataFrame({k: piv[(k, True)] / piv[(k, False)] for k in ['sel6_mean', 'pump_mean', 'position_mean']})
    rel.to_csv(OUT / 'robust_relative.csv')
    print('CB0246 L silenced / intact, per noise seed:\n', rel.round(2).to_string(), flush=True)
    del model

    print('=== B: mirror (right side sugar only) ===', flush=True)
    model = Model(right, p, data=V783)
    run_set(model, {'intact': ([], None), 'CB0246_L': ([CB0246_L], None), 'CB0246_R': ([CB0246_R], None)},
            20, OUT / 'mirror', t_run_s, pump, pos)


if __name__ == '__main__':
    main()
