"""E0: reproduce Shiu et al. 2024 sugar GRN -> MN9 baseline (FlyWire v630)."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, ms  # noqa: E402
from ffsim.sim import run, rates, default_params  # noqa: E402

# right-hemisphere sugar GRNs + MN9, from Shiu example.ipynb (v630 IDs)
NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
ID_MN9 = 720575940660219265

OUT = ROOT / 'results' / 'e0'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', type=float, nargs='+', default=[100, 150, 200])
    ap.add_argument('--n-run', type=int, default=30)
    ap.add_argument('--t-run', type=float, default=1000, help='ms')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    params = dict(default_params)
    params['n_run'] = args.n_run
    params['t_run'] = args.t_run * ms

    summary = []
    for r in args.rates:
        params['r_poi'] = r * Hz
        name = f'sugarR_{int(r)}Hz{args.tag}'
        print(f'>>> {name}')
        df_spk = run(NEU_SUGAR, params)
        df_spk.to_parquet(OUT / f'{name}.parquet')
        df_r = rates(df_spk, args.t_run / 1000, args.n_run)
        df_r.to_csv(OUT / f'{name}_rates.csv')
        mn9 = df_r.loc[ID_MN9] if ID_MN9 in df_r.index else {'r': 0.0, 'std': 0.0}
        summary.append({'exp': name, 'r_poi_Hz': r, 'spikes': len(df_spk),
                        'active_neurons': int((df_r.r > 0).sum()),
                        'MN9_Hz': round(mn9['r'], 1), 'MN9_sd': round(mn9['std'], 1)})

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUT / f'summary{args.tag}.csv', index=False)
    print('\n=== summary ===')
    print(df_sum.to_string(index=False))


if __name__ == '__main__':
    main()
