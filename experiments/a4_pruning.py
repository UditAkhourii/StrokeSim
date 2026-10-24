"""A4: developmental synaptic pruning analogue.

Human development prunes weak synapses (over-pruning ~ schizophrenia hypothesis,
under-pruning ~ autism/epilepsy). Here: remove all connections with fewer than k
synapses ("activity-independent weak-synapse pruning"), versus removing random
connections carrying the same number of synapses (control). For each network:
  - taste:  sugar GRNs 100 Hz -> MN9, SEZ-MA, active size
  - smell:  vinegar ORNs 20 Hz, DA2 ORNs 60 Hz -> runaway? (KC fraction, active size)
Question: does pruning weak synapses remove the olfactory runaway while keeping taste function?
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, ms, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from ffsim.readouts import load_annotations, orn_ids, ODORS  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'a4'
THRESH = [1, 2, 3, 5, 10]   # keep connections with >= k synapses (1 = intact)
RUNAWAY_N = 5000


def networks(syn):
    """name -> per-row weight multiplier (0 = pruned)."""
    total = syn.sum()
    nets = {}
    rng = np.random.default_rng(0)
    for k in THRESH:
        keep = (syn >= k).astype(float)
        nets[f'weak_k{k}'] = keep
        if k > 1:
            removed = total - (syn * keep).sum()
            order = rng.permutation(len(syn))
            cum = np.cumsum(syn[order])
            cut = order[: np.searchsorted(cum, removed) + 1]
            rnd = np.ones(len(syn))
            rnd[cut] = 0.0
            nets[f'random_match_k{k}'] = rnd
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-run', type=int, default=5)
    ap.add_argument('--t-run', type=float, default=500, help='ms')
    args = ap.parse_args()
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)

    a = load_annotations()
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    syn = pd.read_parquet(V783['path_con'], columns=['Connectivity']).Connectivity.values
    sez = list(pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id)
    kc = set(a[a.cell_class == 'Kenyon_Cell'].root_id)
    stims = {'sugar100': ([f for f in NEU_SUGAR if f in ids], 100),
             'vinegar20': (orn_ids(ODORS['vinegar'], a, ids), 20),
             'DA2_60': (orn_ids(['DA2'], a, ids), 60)}

    rows = []
    for net, w in networks(syn).items():
        for st, (exc, hz) in stims.items():
            name = f'{net}__{st}'
            f = OUT / 'rates' / f'{name}.csv'
            if not f.exists():
                print(f'>>> {name} (synapses kept {(syn * w).sum() / syn.sum():.2f})', flush=True)
                p = dict(default_params)
                p.update(r_poi=hz * Hz, n_run=args.n_run, t_run=args.t_run * ms)
                df = run(exc, p, data=V783, w_scale=w, verbose=False)
                rates(df, float(p['t_run'] / second), args.n_run).to_csv(f)
            r = pd.read_csv(f, index_col=0).r
            act = set(r[r > 1].index)
            rows.append({'network': net, 'stim': st, 'syn_kept': (syn * w).sum() / syn.sum(),
                         'edges_kept': w.mean(), 'MN9_Hz': r.get(ID_MN9, 0.0),
                         'SEZ_MA_Hz': r.reindex(sez).fillna(0).mean(), 'active': len(act),
                         'KC_frac': len(act & kc) / len(kc), 'runaway': len(act) > RUNAWAY_N})
            pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
