"""A1b: null distribution for the PRW stroke — random lesions matched to PRW's size.

10 random sets of 400 neurons (neuron:PRW silenced 402), sugar 100 Hz, v783,
10 trials each. Same spare set as A1. Per-neuron rates saved.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'a1b'
N_LES, SEEDS, N_RUN = 400, range(10), 10


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    sez = list(pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id)
    pool = pd.Series(sorted(ids - set(sugar) - set(sez) - {ID_MN9}))
    p = dict(default_params)
    p.update(r_poi=100 * Hz, n_run=N_RUN)
    for s in SEEDS:
        f = OUT / 'rates' / f'random{N_LES}_s{s}.csv'
        if f.exists():
            continue
        les = list(pool.sample(N_LES, random_state=100 + s))
        print(f'>>> seed {s}', flush=True)
        df = run(sugar, p, slnc_ids=les, data=V783, verbose=False)
        rates(df, float(p['t_run'] / second), N_RUN).to_csv(f)


if __name__ == '__main__':
    main()
