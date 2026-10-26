"""K3: does the CB0246 bottleneck survive two-sided sugar input?

The paper (and Shiu et al.'s NEU_SUGAR) stimulates 21 sugar GRNs that are all on the
LEFT side. K1 hits are left-sided (CB0246 L, CB0883 L) while the right twins matter less.
Here sugar is delivered to both sides: the original 21 left GRNs plus 21 right GRNs
matched by cell type (LB3c 9, LB3d 5, LB4b 4, LB3b 2, +1 LB3c for the unannotated one;
lowest IDs per type, deterministic). v783, default W_syn, 100 Hz, 20 trials.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402
from experiments.k2_bilateral import SEL, CB0246_L, CB0246_R, CB0559_L, CB0559_R  # noqa: E402

OUT = ROOT / 'results' / 'k3'
N_RUN = 20
CB0883_L, CB0883_R = 720575940625102692, 720575940629370844
MN9_L = 720575940618238523  # ID_MN9 is the right-side MN9
RIGHT_COUNTS = {'LB3c': 10, 'LB3d': 5, 'LB4b': 4, 'LB3b': 2}
CONDS = {
    'intact': [],
    'CB0246_L': [CB0246_L],
    'CB0246_R': [CB0246_R],
    'CB0246_both': [CB0246_L, CB0246_R],
    'CB0883_L': [CB0883_L],
    'CB0883_both': [CB0883_L, CB0883_R],
    'CB0559_both': [CB0559_L, CB0559_R],
}


def right_sugar(ids):
    a = pd.read_csv(ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv', sep='\t',
                    usecols=['root_id', 'cell_type', 'side'], low_memory=False).drop_duplicates('root_id')
    a = a[(a.side == 'right') & a.root_id.isin(ids)]
    out = []
    for t, k in RIGHT_COUNTS.items():
        out += sorted(a.root_id[a.cell_type == t])[:k]
    return out


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    left = [f for f in NEU_SUGAR if f in ids]
    right = right_sugar(ids)
    print(f'sugar input: {len(left)} left + {len(right)} right GRNs', flush=True)
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    model = Model(left + right, p, data=V783)
    rows = []
    for name, les in CONDS.items():
        f = OUT / 'rates' / f'{name}.csv'
        if not f.exists():
            print(f'>>> {name}', flush=True)
            df, _ = model.run(slnc_ids=les, n_run=N_RUN)
            rates(df, float(p['t_run'] / second), N_RUN).to_csv(f)
        r = pd.read_csv(f, index_col=0).r
        sel = {k: r.get(v, 0.0) for k, v in SEL.items()}
        rows.append({'cond': name, 'MN9_R': r.get(ID_MN9, 0.0), 'MN9_L': r.get(MN9_L, 0.0),
                     'CB0246_L': r.get(CB0246_L, 0.0), 'CB0246_R': r.get(CB0246_R, 0.0),
                     'paper3_mean': sum(v for k, v in sel.items() if k.endswith('*')) / 3,
                     'all6_mean': sum(sel.values()) / 6, **sel})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(d.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
