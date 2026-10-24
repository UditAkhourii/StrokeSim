"""E1: does sugar GRN stimulation drive reward-pathway neurons (PAM DANs, MBONs)?

Sugar spikes come from E0. Bitter GRN stimulation is the specificity control:
punishment DANs (PPL1) should prefer bitter, reward DANs (PAM) should prefer sugar.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
import argparse  # noqa: E402
from ffsim.sim import run, rates, default_params, V630, V783  # noqa: E402
from ffsim.readouts import reward_panel  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR  # noqa: E402

# right-hemisphere bitter GRNs, from Shiu figures.ipynb (v630 IDs)
NEU_BITTER = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
E0 = ROOT / 'results' / 'e0'
OUT = ROOT / 'results' / 'e1'
STIM_HZ = [100, 150]
ACTIVE_HZ = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['630', '783'], default='630')
    args = ap.parse_args()
    data = V630 if args.version == '630' else V783
    out = OUT if args.version == '630' else ROOT / 'results' / 'e1_v783'
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(data['path_comp'], index_col=0)
    ids = set(comp.index)
    stims = {'sugar': [f for f in NEU_SUGAR if f in ids],
             'bitter': [f for f in NEU_BITTER if f in ids]}

    params = dict(default_params)
    cols = {}
    for r in STIM_HZ:
        for stim, exc in stims.items():
            p = out / f'{stim}R_{r}Hz_rates.csv'
            if args.version == '630' and stim == 'sugar':
                p = E0 / f'sugarR_{r}Hz_rates.csv'  # reuse E0
            if not p.exists():
                print(f'>>> {stim}R_{r}Hz ({len(exc)} GRNs, v{args.version})')
                params['r_poi'] = r * Hz
                df_spk = run(exc, params, data=data)
                df_spk.to_parquet(out / f'{stim}R_{r}Hz.parquet')
                rates(df_spk, float(params['t_run'] / second), params['n_run']).to_csv(p)
            cols[f'{stim}_{r}'] = pd.read_csv(p, index_col=0)

    panel = reward_panel(ids=ids)
    for k, df in cols.items():
        panel[k] = panel.flywire_id.map(df['r']).fillna(0.0)
    panel.to_csv(out / 'panel_rates.csv', index=False)

    rate_cols = list(cols)
    grp = panel.groupby('grp')
    n = grp.size().rename('n')
    frac = grp[rate_cols].agg(lambda x: (x > ACTIVE_HZ).mean()).add_prefix('frac_')
    mean = grp[rate_cols].mean().add_prefix('mean_')
    summ = pd.concat([n, frac.round(2), mean.round(2)], axis=1)
    summ.to_csv(out / 'group_summary.csv')
    print('\n=== group summary (frac active >1 Hz, mean Hz) ===')
    print(summ.T.to_string())

    print('\n=== responsive cell types (any condition > 1 Hz) ===')
    by_type = panel.groupby(['grp', 'cell_type'])[rate_cols].mean()
    print(by_type[(by_type > ACTIVE_HZ).any(axis=1)].round(1).to_string())


if __name__ == '__main__':
    main()
