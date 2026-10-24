"""N1: does any global W_syn give sugar -> PAM propagation without runaway activity?

The fox paper (Christie et al. 2025) raises W_syn to make PAMs fire. Here we
check what the same raise does to network stability. v783, sugar or bitter
GRNs at 100 Hz. As in Shiu's code, the Poisson input weight is w_syn * f_poi,
so it scales with W_syn.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, mV, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from ffsim.readouts import load_annotations  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402
from experiments.e1_reward_readout import NEU_BITTER  # noqa: E402

OUT = ROOT / 'results' / 'n1'
W_SYN = [0.275, 0.35, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.7, 0.9]
N_RUN = 10
CB0233 = [720575940617648453, 720575940655011489]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = load_annotations()
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    ct = a.set_index('root_id').cell_type.astype(str)
    cls = a.set_index('root_id').cell_class
    groups = {
        'PAM': set(ct[ct.str.startswith('PAM')].index),
        'PPL1': set(ct[ct.str.startswith('PPL1')].index),
        'MBON': set(cls[cls == 'MBON'].index),
        'KC': set(cls[cls == 'Kenyon_Cell'].index),
        'ALPN': set(cls[cls == 'ALPN'].index),
        'SEZ_DA': set(pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id),
    }
    stims = {'sugar': [f for f in NEU_SUGAR if f in ids], 'bitter': [f for f in NEU_BITTER if f in ids]}
    rows = []
    for w in W_SYN:
        for stim, exc in stims.items():
            name = f'{stim}_w{w}'
            p_csv = OUT / f'{name}_rates.csv'
            if not p_csv.exists():
                print(f'>>> {name}', flush=True)
                params = dict(default_params)
                params.update(w_syn=w * mV, r_poi=100 * Hz, n_run=N_RUN)
                df = run(exc, params, data=V783)
                rates(df, float(params['t_run'] / second), N_RUN).to_csv(p_csv)
            r = pd.read_csv(p_csv, index_col=0).r
            act = set(r[r > 1.0].index)
            row = {'stim': stim, 'w_syn_mV': w, 'active_total': len(act),
                   'MN9_Hz': round(r.get(ID_MN9, 0.0), 1),
                   'CB0233_Hz': round(r.reindex(CB0233).fillna(0).max(), 1)}
            for g, s in groups.items():
                row[f'{g}_frac'] = round(len(act & s) / len(s), 3)
            row['SEZ_DA_Hz'] = round(r.reindex(list(groups['SEZ_DA'])).fillna(0).mean(), 1)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
