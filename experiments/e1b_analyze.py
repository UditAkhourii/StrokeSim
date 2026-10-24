"""E1b analysis: option 1 (SEZ dopamine readout) and option 2 (PAM subthreshold Vm).

Run after e1b_readouts.py. Writes results/e1b/summary_*.csv and figures.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ffsim.readouts import load_annotations  # noqa: E402

E1B = ROOT / 'results' / 'e1b'
SUGAR = [20, 50, 100, 150, 200]
ACTIVE_HZ = 1.0


def load(name):
    r = pd.read_csv(E1B / f'{name}_rates.csv', index_col=0).r
    v = pd.read_csv(E1B / f'{name}_dv.csv', index_col=0)
    return r, v


def main():
    ann = load_annotations().set_index('root_id')
    conds = sorted(p.name[:-len('_dv.csv')] for p in E1B.glob('*_dv.csv'))
    data = {c: load(c) for c in conds}

    # ---- option 1: SEZ dopamine readout --------------------------------
    # definition: predicted-dopaminergic, not MB DAN, >1 Hz at sugar_150 and <1 Hz at bitter_100
    r150, rbit = data['sugar_150'][0], data['bitter_100'][0]
    da = ann[(ann.top_nt == 'dopamine') & (ann.cell_class != 'DAN')].index
    sel = [f for f in da if r150.get(f, 0) > ACTIVE_HZ and rbit.get(f, 0) <= ACTIVE_HZ]
    pd.DataFrame({'flywire_id': sel}).to_csv(E1B / 'sez_da_ids.csv', index=False)
    tab = ann.loc[sel, ['cell_type', 'side', 'super_class', 'top_nt_conf']].copy()
    for c in conds:
        tab[c] = [data[c][0].get(f, 0.0) for f in sel]
    tab.round(1).to_csv(E1B / 'summary_sez_da_neurons.csv')
    print(f'=== option 1: SEZ_DA set = {len(sel)} neurons ===')
    print(tab[['cell_type', 'side', 'top_nt_conf'] + conds].round(1).to_string())

    # ---- option 2: subthreshold Vm per group ---------------------------
    grp = pd.Series('other', index=ann.index)
    ct = ann.cell_type.astype(str)
    grp[ct.str.startswith('PAM')] = 'PAM'
    grp[ct.str.startswith('PPL1')] = 'PPL1'
    grp[ann.cell_class == 'MBON'] = 'MBON'
    grp[ann.cell_class == 'Kenyon_Cell'] = 'KC'
    grp[ann.cell_class == 'ALPN'] = 'PN'
    grp[sel] = 'SEZ_DA'
    rows = []
    for c in conds:
        r, v = data[c]
        g = grp.reindex(v.index)
        for k, vv in v.groupby(g):
            rows.append({'cond': c, 'grp': k, 'n': len(vv),
                         'dv_mean': vv['mean'].mean(), 'dv_max': vv['mean'].max(),
                         'frac_dv_gt_0.1mV': (vv['mean'] > 0.1).mean(),
                         'trial_sd': vv['std'].mean(),
                         'frac_spiking': (r.reindex(vv.index).fillna(0) > ACTIVE_HZ).mean(),
                         'rate_mean': r.reindex(vv.index).fillna(0).mean()})
    s = pd.DataFrame(rows)
    s.to_csv(E1B / 'summary_groups.csv', index=False)
    piv = s.pivot(index='cond', columns='grp', values='dv_mean').round(3)
    print('\n=== option 2: mean depolarisation above rest (mV) ===')
    print(piv.to_string())
    print('\n=== frac of group with dv > 0.1 mV ===')
    print(s.pivot(index='cond', columns='grp', values='frac_dv_gt_0.1mV').round(2).to_string())

    # PAM per-neuron dose-response: graded? signal vs trial noise?
    pam = [f for f in data['sugar_150'][1].index if grp.get(f) == 'PAM']
    dv = pd.DataFrame({r: data[f'sugar_{r}'][1].loc[pam, 'mean'] for r in SUGAR})
    sd = pd.DataFrame({r: data[f'sugar_{r}'][1].loc[pam, 'std'] for r in SUGAR})
    sem = sd / np.sqrt(30)
    resp = dv[150] > 3 * sem[150]
    print(f'\nPAM responsive at sugar_150 (dv > 3 SEM): {resp.sum()}/{len(pam)}')
    if resp.any():
        mono = (dv[resp].diff(axis=1).iloc[:, 1:] >= -1e-3).all(axis=1).mean()
        print(f'  of those, monotonic in sugar rate: {mono:.2f}')
        print(dv[resp].join(ann.cell_type).sort_values(150, ascending=False).head(12).round(3).to_string())
    dv.join(ann.cell_type).to_csv(E1B / 'summary_pam_dv_dose.csv')

    # ---- figure -------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    sez = s[(s.grp == 'SEZ_DA') & s.cond.str.startswith('sugar')].copy()
    sez['hz'] = sez.cond.str.split('_').str[1].astype(int)
    sez = sez.sort_values('hz')
    ax[0].plot(sez.hz, sez.rate_mean, 'o-', color='#2a6f97', label='sugar')
    ax[0].axhline(s[(s.grp == 'SEZ_DA') & (s.cond == 'bitter_100')].rate_mean.iloc[0],
                  color='#a44a3f', ls='--', label='bitter 100 Hz')
    ax[0].set(xlabel='Sugar GRN stim (Hz)', ylabel='mean rate (Hz)', title=f'Option 1: SEZ_DA (n={len(sel)})')
    ax[0].legend(frameon=False)
    for k, col in [('PAM', '#2a6f97'), ('PPL1', '#a44a3f'), ('MBON', '#6a994e')]:
        d = s[(s.grp == k) & s.cond.str.startswith('sugar')].copy()
        d['hz'] = d.cond.str.split('_').str[1].astype(int)
        d = d.sort_values('hz')
        ax[1].plot(d.hz, d.dv_mean, 'o-', color=col, label=k)
    ax[1].set(xlabel='Sugar GRN stim (Hz)', ylabel='mean Δv above rest (mV)',
              title='Option 2: subthreshold depolarisation')
    ax[1].legend(frameon=False)
    ax[2].hist(dv[150], bins=40, color='#2a6f97')
    ax[2].set(xlabel='PAM Δv at sugar 150 Hz (mV)', ylabel='# PAM neurons', title='PAM Δv distribution')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(E1B / 'e1b_readouts.png', dpi=150)


if __name__ == '__main__':
    main()
