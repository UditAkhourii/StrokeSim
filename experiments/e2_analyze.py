"""E2 analysis: lesion atlas summary, dissociation (DLI vs MN9), size controls, figures."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / 'results' / 'e2'


def z(x, sd, n, x0, sd0, n0):
    se = np.sqrt(sd ** 2 / n + sd0 ** 2 / n0)
    return (x - x0) / np.where(se > 0, se, np.nan)


def main():
    df = pd.concat([pd.read_csv(p) for p in sorted((E2 / 'conds').glob('*.csv'))], ignore_index=True)
    base = df[df.cond == 'intact'].iloc[0]
    df = df[df.cond != 'intact'].copy()
    df['mn9_ratio'] = df.mn9 / base.mn9
    df['dli'] = df.sez_da / base.sez_da                      # depression-like index v2
    df['pam_dv_uV'] = df.pam_dv * 1000
    df['mn9_z'] = z(df.mn9, df.mn9_sd, df.n_run, base.mn9, base.mn9_sd, base.n_run)
    df['sez_z'] = z(df.sez_da, df.sez_da_sd, df.n_run, base.sez_da, base.sez_da_sd, base.n_run)
    df['dissoc'] = df.dli - df.mn9_ratio                     # <0: reward lost beyond motor
    df.to_csv(E2 / 'atlas_summary.csv', index=False)

    # ---- hit screen: outside the random-control range AND |z| > 3 vs intact ----
    # SEZ_DA is relabelled SEZ-MA (3/9 are immuno-confirmed serotonergic; not a clean DA index)
    rnd = df[df.type == 'random']
    hits = []
    if len(rnd):
        for metric, zcol in [('mn9_ratio', 'mn9_z'), ('dli', 'sez_z'), ('pam_dv_uV', None)]:
            lo, hi = rnd[metric].min(), rnd[metric].max()
            out = (df[metric] < lo) | (df[metric] > hi)
            if zcol:
                out &= df[zcol].abs() > 3
            for c in df[out & (df.type != 'random')].cond:
                hits.append((c, metric))
        h = pd.DataFrame(hits, columns=['cond', 'metric'])
        h = h.groupby('cond').metric.apply(lambda x: ','.join(sorted(set(x)))).reset_index()
        h.to_csv(E2 / 'hits.csv', index=False)
        print('random-control ranges:', {m: (round(rnd[m].min(), 3), round(rnd[m].max(), 3))
                                         for m in ['mn9_ratio', 'dli', 'pam_dv_uV']})
        print(f'\n=== HITS ({len(h)}) ===')
        print(h.merge(df, on='cond')[['cond', 'metric', 'n_run', 'n_silenced', 'mn9_ratio', 'dli',
                                      'pam_dv_uV', 'active_neurons']].round(2).to_string(index=False))

    print(f"intact: MN9 {base.mn9:.1f} Hz, SEZ_DA {base.sez_da:.1f} Hz, PAM dv {base.pam_dv*1000:.2f} uV, "
          f"active {base.active_neurons}")
    cols = ['cond', 'n_silenced', 'syn_removed', 'mn9_ratio', 'dli', 'dissoc', 'pam_dv_uV', 'active_neurons']
    for t in ['tissue', 'neuron']:
        d = df[df.type == t]
        hit = d[(d.mn9_ratio < 0.9) | (d.dli < 0.9) | (d.mn9_ratio > 1.1) | (d.dli > 1.1)]
        print(f'\n=== {t} lesions: {len(d)} total, {len(hit)} change MN9 or SEZ_DA by >10% ===')
        print(hit.sort_values('dli')[cols].round(2).to_string(index=False))
        if t == 'tissue':
            rho, p = spearmanr(d.syn_removed, 1 - d.dli)
            print(f'  size vs DLI loss: Spearman rho={rho:.2f} p={p:.2g}')
            rho, p = spearmanr(d.syn_removed, 1 - d.mn9_ratio)
            print(f'  size vs MN9 loss: Spearman rho={rho:.2f} p={p:.2g}')

    rnd = df[df.type == 'random'].sort_values('n_silenced')
    print('\n=== random size controls ===')
    print(rnd[cols].round(2).to_string(index=False))

    # anhedonia-like: reward readout lost, motor spared (both significant vs noise)
    anh = df[(df.dli < 0.7) & (df.mn9_ratio > 0.85) & (df.sez_z < -3)]
    print('\n=== anhedonia-like lesions (DLI<0.7, MN9>0.85) ===')
    print(anh[cols].round(2).to_string(index=False) if len(anh) else '  none')
    mot = df[(df.mn9_ratio < 0.7) & (df.dli > 0.85)]
    print('\n=== motor-only lesions (MN9<0.7, DLI>0.85) ===')
    print(mot[cols].round(2).to_string(index=False) if len(mot) else '  none')

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for t, col, mk in [('tissue', '#2a6f97', 'o'), ('neuron', '#a44a3f', 's'), ('random', '#888888', '^')]:
        d = df[df.type == t]
        ax[0].scatter(d.mn9_ratio, d.dli, s=22, c=col, marker=mk, label=t, alpha=.8)
    lab = df[(df.type != 'random') & ((df.mn9_ratio < 0.85) | (df.dli < 0.85) | (df.dli > 1.15))]
    for _, r in lab.iterrows():
        ax[0].annotate(r.cond.replace('tissue:', 't:').replace('neuron:', 'n:'), (r.mn9_ratio, r.dli),
                       fontsize=7, xytext=(3, 2), textcoords='offset points')
    lim = [-0.05, max(1.3, df[['mn9_ratio', 'dli']].max().max() + .05)]
    ax[0].plot(lim, lim, color='#ccc', lw=1, zorder=0)
    ax[0].axvline(1, color='#eee', lw=1, zorder=0); ax[0].axhline(1, color='#eee', lw=1, zorder=0)
    ax[0].set(xlim=lim, ylim=lim, xlabel='MN9 (motor) / intact', ylabel='SEZ-MA / intact',
              title='Lesion effects: reward vs motor')
    ax[0].legend(frameon=False, fontsize=8)

    d = df[df.type == 'tissue'].copy()
    d['worst'] = d[['mn9_ratio', 'dli']].min(axis=1)
    d = d[(d.worst < 0.95) | (d.dli > 1.05) | (d.mn9_ratio > 1.05)].sort_values('worst')
    y = np.arange(len(d))
    ax[1].barh(y - .2, d.mn9_ratio, .4, color='#a44a3f', label='MN9')
    ax[1].barh(y + .2, d.dli, .4, color='#2a6f97', label='SEZ-MA')
    ax[1].set_yticks(y, d.region, fontsize=7); ax[1].invert_yaxis(); ax[1].axvline(1, color='#999', lw=1)
    ax[1].set(xlabel='readout / intact', title='Tissue lesions with >5% effect'); ax[1].legend(frameon=False, fontsize=8)

    for t, col, mk in [('neuron', '#a44a3f', 's'), ('random', '#888888', '^')]:
        d = df[df.type == t]
        ax[2].scatter(d.n_silenced, 1 - d.dli, s=22, c=col, marker=mk, label=f'{t}: SEZ-MA loss')
        ax[2].scatter(d.n_silenced, 1 - d.mn9_ratio, s=22, facecolors='none', edgecolors=col, marker=mk,
                      label=f'{t}: MN9 loss')
    ax[2].set(xscale='log', xlabel='# neurons silenced', ylabel='fractional loss',
              title='Neuron lesions vs random size controls')
    ax[2].legend(frameon=False, fontsize=7)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(E2 / 'e2_atlas.png', dpi=150)


if __name__ == '__main__':
    main()
