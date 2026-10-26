"""K1: single-neuron knockout screen over the PRW lesion set.

Is CB0246 the only PRW neuron whose silencing removes the serotonergic SEZ response?
Silence each of the PRW neurons (same set as the PRW neuron stroke) one at a time,
sugar 100 Hz, v783, default W_syn. Network built once (ffsim.sim.Model).

Screen: --n-run trials per neuron (default 5) plus an intact baseline (--n-intact).
Confirm: re-run neurons flagged in the screen with --confirm (default 20 trials),
written to results/k1/confirm/. A neuron is a hit when the 5-HT mean falls below
--hit-frac of intact (default 0.5). MN9 is reported alongside.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from ffsim.lesion import neuropil_lesion  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'k1'
HT = [720575940636895406, 720575940631179831, 720575940628676457]
CB0246 = 720575940632602532


def readout(r):
    ht = [r.get(h, 0.0) for h in HT]
    return {'MN9_Hz': r.get(ID_MN9, 0.0), '5HT_mean_Hz': sum(ht) / 3,
            **{f'5HT_{i}': v for i, v in enumerate(ht)}, 'CB0246_Hz': r.get(CB0246, 0.0)}


def sweep(model, conds, n_run, outdir, t_run_s):
    """conds: {name: [ids to silence]}. Skips conditions already on disk."""
    (outdir / 'rates').mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (name, les) in enumerate(conds.items()):
        f = outdir / 'rates' / f'{name}.csv'
        if not f.exists():
            df, _ = model.run(slnc_ids=les, n_run=n_run)
            rates(df, t_run_s, n_run).to_csv(f)
        r = pd.read_csv(f, index_col=0).r
        rows.append({'cond': name, **readout(r)})
        if (k + 1) % 10 == 0 or k + 1 == len(conds):
            print(f'    {k + 1}/{len(conds)}', flush=True)
            pd.DataFrame(rows).to_csv(outdir / 'summary.csv', index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-run', type=int, default=5)
    ap.add_argument('--n-intact', type=int, default=20)
    ap.add_argument('--confirm', type=int, default=20)
    ap.add_argument('--hit-frac', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=None, help='first N neurons only (smoke test)')
    a = ap.parse_args()

    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    prw = sorted(f for f in neuropil_lesion('PRW', spare=set(sugar) | set(HT) | {ID_MN9}) if f in ids)
    if CB0246 not in prw:
        prw.append(CB0246)  # positive control
    prw = prw[:a.limit] if a.limit else prw
    print(f'PRW knockout set: {len(prw)} neurons (CB0246 included)', flush=True)

    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    t_run_s = float(p['t_run'] / second)
    print('>>> building v783 network', flush=True)
    model = Model(sugar, p, data=V783)

    print('>>> intact', flush=True)
    intact = sweep(model, {'intact': []}, a.n_intact, OUT, t_run_s).iloc[0]
    print('>>> screen', flush=True)
    scr = sweep(model, {'intact': [], **{str(f): [f] for f in prw}}, a.n_run, OUT / 'screen', t_run_s)
    scr = scr[scr.cond != 'intact']
    base = intact['5HT_mean_Hz']
    scr['5HT_rel'] = scr['5HT_mean_Hz'] / base
    scr['MN9_rel'] = scr['MN9_Hz'] / intact['MN9_Hz']
    scr.sort_values('5HT_rel').to_csv(OUT / 'screen_ranked.csv', index=False)
    hits = scr[scr['5HT_rel'] < a.hit_frac].cond.tolist()
    print(f'intact: MN9 {intact.MN9_Hz:.1f} Hz, 5-HT mean {base:.2f} Hz; '
          f'{len(hits)} screen hits (< {a.hit_frac:.0%} of intact)', flush=True)

    if hits and a.confirm:
        print('>>> confirm', flush=True)
        conf = sweep(model, {'intact': [], **{h: [int(h)] for h in hits}}, a.confirm,
                     OUT / 'confirm', t_run_s)
        print(conf.round(2).to_string(index=False))
    print(scr.sort_values('5HT_rel').head(15).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
