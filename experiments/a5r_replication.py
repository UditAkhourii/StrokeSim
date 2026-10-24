"""Replication of the PRW / CB0246 stroke findings.

Part 1, robustness: for each setting, run intact, PRW neuron stroke, CB0246 alone,
and 3 random 400-neuron strokes. Settings: v783 at W_syn 0.25/0.30/0.35 mV, and
v630 at the default 0.275 mV (IDs absent in v630 dropped; 2 of 3 5-HT cells present).

Part 2, round-stroke null: 10 random strokes matched to the PRW_r60 + penumbra
sphere (935 neurons silenced + 3711 at 50% weight), v783, default W_syn.
Sugar 100 Hz, 10 trials each; per-neuron rates saved.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, mV, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V630, V783  # noqa: E402
from ffsim.lesion import neuropil_lesion  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'replication'
HT = [720575940636895406, 720575940631179831, 720575940628676457]
CB0246 = 720575940632602532
N_RUN = 10
SETTINGS = [('v783_w0.25', V783, 0.25), ('v783_w0.30', V783, 0.30),
            ('v783_w0.35', V783, 0.35), ('v630_w0.275', V630, 0.275)]


def one(name, data, w, exc, slnc=(), scale=(), factor=1.0):
    f = OUT / 'rates' / f'{name}.csv'
    if not f.exists():
        print(f'>>> {name}', flush=True)
        p = dict(default_params)
        p.update(r_poi=100 * Hz, n_run=N_RUN, w_syn=w * mV)
        df = run(exc, p, slnc_ids=list(slnc), data=data, verbose=False,
                 scale_ids=list(scale), scale_factor=factor)
        rates(df, float(p['t_run'] / second), N_RUN).to_csv(f)
    r = pd.read_csv(f, index_col=0).r
    return {'cond': name, 'MN9_Hz': r.get(ID_MN9, 0.0),
            '5HT_mean_Hz': r.reindex(HT).dropna().mean() if r.reindex(HT).notna().any() else 0.0,
            **{f'5HT_{i}': r.get(h, 0.0) for i, h in enumerate(HT)}, 'CB0246_Hz': r.get(CB0246, 0.0)}


def main():
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, data, w in SETTINGS:
        ids = set(pd.read_csv(data['path_comp'], index_col=0).index)
        sugar = [f for f in NEU_SUGAR if f in ids]
        prw = [f for f in neuropil_lesion('PRW', spare=set(sugar) | set(HT) | {ID_MN9}) if f in ids]
        pool = pd.Series(sorted(ids - set(sugar) - set(HT) - {ID_MN9}))
        rows.append({'setting': tag, 'lesion': 'intact', **one(f'{tag}__intact', data, w, sugar)})
        rows.append({'setting': tag, 'lesion': 'PRW', **one(f'{tag}__PRW', data, w, sugar, prw)})
        rows.append({'setting': tag, 'lesion': 'CB0246', **one(f'{tag}__CB0246', data, w, sugar, [CB0246])})
        for s in range(3):
            les = pool.sample(400, random_state=200 + s)
            rows.append({'setting': tag, 'lesion': f'random400_s{s}', **one(f'{tag}__rand400_s{s}', data, w, sugar, les)})
        pd.DataFrame(rows).to_csv(OUT / 'robustness.csv', index=False)

    # Part 2: round-stroke null (v783 default)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    pool = pd.Series(sorted(ids - set(sugar) - set(HT) - {ID_MN9}))
    null = []
    for s in range(10):
        pick = pool.sample(935 + 3711, random_state=300 + s)
        null.append({'seed': s, **one(f'roundnull_s{s}', V783, 0.275, sugar,
                                     pick[:935], pick[935:], 0.5)})
        pd.DataFrame(null).to_csv(OUT / 'round_stroke_null.csv', index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print(pd.DataFrame(null).round(1).to_string(index=False))


if __name__ == '__main__':
    main()
