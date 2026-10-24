"""A3: is CB0246 the single-neuron bottleneck for the PRW-stroke serotonin loss?

Silence CB0246 alone (and the next-strongest PRW inputs, as controls), sugar 100 Hz,
v783, 10 trials. Readouts: MN9 and the 3 serotonergic SEZ cells.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'a3'
HT = [720575940636895406, 720575940631179831, 720575940628676457]
CONDS = {
    'CB0246': [720575940632602532],
    'CB0208': [720575940610611246],
    'CB1366_x2': [720575940630664556, 720575940627312668],
}


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    p = dict(default_params)
    p.update(r_poi=100 * Hz, n_run=10)
    rows = []
    for name, les in CONDS.items():
        f = OUT / 'rates' / f'{name}.csv'
        if not f.exists():
            print(f'>>> silence {name}', flush=True)
            df = run(sugar, p, slnc_ids=les, data=V783, verbose=False)
            rates(df, float(p['t_run'] / second), 10).to_csv(f)
        r = pd.read_csv(f, index_col=0).r
        rows.append({'silenced': name, 'MN9_Hz': round(r.get(ID_MN9, 0), 1),
                     **{f'5HT_{i}': round(r.get(h, 0), 1) for i, h in enumerate(HT)}})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'summary.csv', index=False)
    print(d.to_string(index=False))


if __name__ == '__main__':
    main()
