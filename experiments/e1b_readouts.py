"""E1b: alternative reward readouts after E1 found PAM/MBONs silent.

Conditions (v783, 30 trials x 1 s), all recording Vm of the readout panel:
  sugar GRNs at 20-200 Hz, bitter GRNs at 100 Hz, ORN odour stand-ins at 100 Hz.
Outputs per condition: <name>_rates.csv (all spiking neurons), <name>_dv.csv (panel Vm).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from ffsim.readouts import load_annotations, reward_panel, olfactory_panel, orn_ids, ODORS  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR  # noqa: E402
from experiments.e1_reward_readout import NEU_BITTER  # noqa: E402

OUT = ROOT / 'results' / 'e1b'


def conditions(ann, ids):
    sugar = [f for f in NEU_SUGAR if f in ids]
    bitter = [f for f in NEU_BITTER if f in ids]
    conds = {f'sugar_{r}': (sugar, r) for r in [20, 50, 100, 150, 200]}
    conds['bitter_100'] = (bitter, 100)
    for k, g in ODORS.items():
        conds[f'odor_{k}_100'] = (orn_ids(g, ann, ids), 100)
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='subset of condition names')
    ap.add_argument('--n-run', type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ann = load_annotations()
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    # record Vm of every DAN/MBON, every predicted-dopaminergic neuron, APL and a KC sample
    da = ann[(ann.top_nt == 'dopamine') & ann.root_id.isin(ids)].root_id
    olf = olfactory_panel(ann, ids)
    kc_sample = olf[olf.grp == 'KC'].flywire_id.sample(300, random_state=0)
    rec = pd.unique(pd.concat([reward_panel(ann, ids).flywire_id, da,
                               olf[olf.grp.isin(['APL', 'PN'])].flywire_id, kc_sample]))

    params = dict(default_params)
    params['n_run'] = args.n_run
    for name, (exc, r) in conditions(ann, ids).items():
        if args.only and name not in args.only:
            continue
        if (OUT / f'{name}_dv.csv').exists():
            continue
        print(f'>>> {name}: {len(exc)} input neurons at {r} Hz', flush=True)
        params['r_poi'] = r * Hz
        df_spk, df_v = run(exc, params, data=V783, record_v=list(rec))
        rates(df_spk, float(params['t_run'] / second), params['n_run']).to_csv(OUT / f'{name}_rates.csv')
        df_v.groupby('flywire_id').dv_mV.agg(['mean', 'std']).to_csv(OUT / f'{name}_dv.csv')


if __name__ == '__main__':
    main()
