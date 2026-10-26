"""K2: is the CB0246 bottleneck really a left/right pair?

The paper silences only CB0246 L (720575940632602532, 51.0% of synapses in PRW). Its mirror
twin CB0246 R (720575940620170887, 49.9% in PRW) falls under the 50% neuropil cutoff and
was never tested. K1 found both CB0559 cells (inputs to CB0246) as screen hits.

Readout widened from the paper's 3 cells to all 6 Yao and Scott 2022 immuno-serotonergic
SEZ cells of the candidate sugar-SEL types (CB0262, CB0892, CB0212; L and R).
v783, default W_syn, sugar 100 Hz, 20 trials per condition.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'k2'
N_RUN = 20
CB0246_L, CB0246_R = 720575940632602532, 720575940620170887
CB0559_L, CB0559_R = 720575940619473624, 720575940620780272
SEL = {  # Yao and Scott 2022 immuno-5HT cells; * = in the paper's readout
    'CB0262_L*': 720575940636895406, 'CB0262_R': 720575940635239672,
    'CB0892_L*': 720575940628676457, 'CB0892_R*': 720575940631179831,
    'CB0212_L': 720575940612289010, 'CB0212_R': 720575940622474899,
}
CONDS = {
    'intact': [],
    'CB0246_L': [CB0246_L],
    'CB0246_R': [CB0246_R],
    'CB0246_both': [CB0246_L, CB0246_R],
    'CB0559_both': [CB0559_L, CB0559_R],
}


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    model = Model(sugar, p, data=V783)
    rows = []
    for name, les in CONDS.items():
        f = OUT / 'rates' / f'{name}.csv'
        if not f.exists():
            print(f'>>> {name}', flush=True)
            df, _ = model.run(slnc_ids=les, n_run=N_RUN)
            rates(df, float(p['t_run'] / second), N_RUN).to_csv(f)
        r = pd.read_csv(f, index_col=0).r
        sel = {k: r.get(v, 0.0) for k, v in SEL.items()}
        rows.append({'cond': name, 'MN9': r.get(ID_MN9, 0.0),
                     'CB0246_L': r.get(CB0246_L, 0.0), 'CB0246_R': r.get(CB0246_R, 0.0),
                     'paper3_mean': sum(v for k, v in sel.items() if k.endswith('*')) / 3,
                     'all6_mean': sum(sel.values()) / 6, **sel})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(d.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
