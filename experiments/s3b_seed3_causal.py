"""S3b: causal test of the seed-3 round-stroke null.

S3 found seed 3 is the only round null that hits CB0246 (penumbra, 50% outgoing weight).
If that explains its near-total 5-HT loss, sparing CB0246 should rescue the response and
halving CB0246 alone should reproduce it. v783, default W_syn, sugar 100 Hz.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402
from experiments.k1_knockout_screen import readout, CB0246, HT  # noqa: E402
from experiments.s3_dissect_roundnull import round_null  # noqa: E402

OUT = ROOT / 'results' / 's3b'
N_RUN = 20


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    slnc, pen = round_null(3, ids)
    assert CB0246 in pen
    conds = {  # name: (silenced, halved)
        'intact': ([], []),
        'seed3': (sorted(slnc), sorted(pen)),
        'seed3_spare_CB0246': (sorted(slnc), sorted(pen - {CB0246})),
        'CB0246_half': ([], [CB0246]),
        'CB0246_silenced': ([CB0246], []),
    }
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    model = Model(sugar, p, data=V783)
    rows = []
    for name, (s, h) in conds.items():
        f = OUT / 'rates' / f'{name}.csv'
        if not f.exists():
            print(f'>>> {name}', flush=True)
            df, _ = model.run(slnc_ids=s, scale_ids=h, scale_factor=0.5, n_run=N_RUN)
            rates(df, float(p['t_run'] / second), N_RUN).to_csv(f)
        r = pd.read_csv(f, index_col=0)
        rows.append({'cond': name, **readout(r.r),
                     '5HT_sd_mean': r['std'].reindex(HT).fillna(0).mean()})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'summary.csv', index=False)
    print(d.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
