"""Main-text figures for 'A single relay neuron decides the serotonergic cost of stroke
in a whole-brain fly model'. Reads results/ only (no simulation).

usage: python experiments/paper_figures.py OUT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
MN9 = 720575940660219265
HT = [720575940636895406, 720575940631179831, 720575940628676457]  # CB0262, CB0892 R, CB0892 L
HT_LBL = ['CB0262', 'CB0892-R', 'CB0892-L']
C = {'neuron': '#a44a3f', 'tissue': '#2a6f97', 'random': '0.65', 'hit': '#e09f3e'}

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def rd(p):
    return pd.read_csv(p, index_col=0).r


def ht(r):
    return r.reindex(HT).fillna(0).mean()


def fig_atlas(out):
    """Fig 3: lesion atlas, feeding vs serotonergic signal."""
    base = rd(R / 'a1' / 'rates' / 'INTACT.csv')
    b_mn9, b_ht = base.get(MN9, 0), ht(base)
    rows = []
    for p in (R / 'a1' / 'rates').glob('*.csv'):
        if p.stem == 'INTACT':
            continue
        r = rd(p)
        kind = 'random' if p.stem.startswith('RANDOM') else 'neuron'
        rows.append((kind, p.stem, r.get(MN9, 0) / b_mn9, ht(r) / b_ht))
    for p in (R / 'a1b' / 'rates').glob('*.csv'):
        r = rd(p)
        rows.append(('random', p.stem, r.get(MN9, 0) / b_mn9, ht(r) / b_ht))
    t = pd.read_csv(R / 'e2' / 'tissue_screen.csv')
    e2b = rd(R / 'e2' / 'rates' / 'intact.csv')
    t_ht = t[[f'sez_{h}' for h in HT]].mean(axis=1) / ht(e2b)
    for _, x in t.iterrows():
        rows.append(('tissue', x.region, x.MI, t_ht[_]))
    d = pd.DataFrame(rows, columns=['kind', 'name', 'MI', 'HT'])
    d.to_csv(out / 'fig3_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for k in ['random', 'tissue', 'neuron']:
        g = d[d.kind == k]
        ax.scatter(g.MI, g.HT, s=22 if k != 'random' else 18, color=C[k], alpha=0.85,
                   label={'neuron': 'neuron lesion (51 neuropils)', 'tissue': 'tissue lesion (67 neuropils)',
                          'random': 'random size-matched (18)'}[k], edgecolor='none')
    for _, x in d[d.name.isin(['GNG', 'PRW'])].iterrows():
        dy = 14 if x.kind == 'neuron' else -12
        ax.annotate(f'{x["name"]} ({x.kind})', (x.MI, x.HT), xytext=(-60 if x['name'] == 'PRW' else 6, dy),
                    textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5))
    ax.axhline(1, ls=':', color='0.6'); ax.axvline(1, ls=':', color='0.6')
    ax.set(xlabel='Feeding output (MN9 / intact)', ylabel='Serotonergic SEZ signal (3 cells / intact)',
           xlim=(-0.05, 1.3), ylim=(-0.05, 1.6), title='Stroke atlas (sugar 100 Hz, 5 trials)')
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(out / 'fig3_stroke_atlas.png', dpi=200); plt.close(fig)


def fig_bottleneck(out):
    """Fig 4: PRW / CB0246 — per-cell serotonin rates and MN9."""
    conds = [('Intact', R / 'e2_confirm' / 'rates' / 'intact.csv'),
             ('PRW tissue', R / 'e2_confirm' / 'rates' / 'tissue__PRW.csv'),
             ('PRW neuron', R / 'e2_confirm' / 'rates' / 'neuron__PRW.csv'),
             ('CB0246 only', R / 'a3' / 'rates' / 'CB0246.csv'),
             ('CB0208 only', R / 'a3' / 'rates' / 'CB0208.csv'),
             ('CB1366 x2', R / 'a3' / 'rates' / 'CB1366_x2.csv')]
    conds = [(n, p) for n, p in conds if p.exists()]
    rnd = [rd(p) for p in sorted((R / 'a1b' / 'rates').glob('*.csv'))]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={'width_ratios': [2.2, 1]})
    x = np.arange(len(conds) + 1)
    w = 0.26
    for i, h in enumerate(HT):
        vals = [rd(p).get(h, 0) for _, p in conds] + [np.mean([r.get(h, 0) for r in rnd])]
        err = [0] * len(conds) + [np.std([r.get(h, 0) for r in rnd])]
        ax[0].bar(x + (i - 1) * w, vals, w, yerr=err, label=HT_LBL[i], color=['#5e548e', '#9f86c0', '#be95c4'][i])
        for xi, v in zip(x, vals):
            if v < 0.05:
                ax[0].text(xi + (i - 1) * w, 0.15, '0', ha='center', fontsize=7, color='#5e548e')
    ax[0].set_xticks(x, [n for n, _ in conds] + ['Random 400\n(n=10)'], rotation=20)
    ax[0].set(ylabel='Firing rate (Hz)', title='Serotonergic SEZ cells')
    ax[0].legend(frameon=False, fontsize=7)
    mn = [rd(p).get(MN9, 0) for _, p in conds] + [np.mean([r.get(MN9, 0) for r in rnd])]
    ax[1].barh(x, mn, color='0.55')
    ax[1].set_yticks(x, [n for n, _ in conds] + ['Random 400'])
    ax[1].invert_yaxis()
    ax[1].set(xlabel='MN9 (Hz)', title='Feeding output')
    fig.tight_layout(); fig.savefig(out / 'fig4_bottleneck.png', dpi=200); plt.close(fig)


def fig_replication(out):
    """Fig 5: replication across connectome version and W_syn."""
    d = pd.read_csv(R / 'replication' / 'robustness.csv')
    d['HT'] = d[['5HT_0', '5HT_1', '5HT_2']].replace(0, np.nan).mean(axis=1)
    base = d[d.lesion == 'intact'].set_index('setting')
    d['HT_rel'] = d.apply(lambda x: (x[['5HT_0', '5HT_1', '5HT_2']].sum()) /
                          base.loc[x.setting, ['5HT_0', '5HT_1', '5HT_2']].sum(), axis=1)
    d['MN9_rel'] = d.apply(lambda x: x.MN9_Hz / base.loc[x.setting, 'MN9_Hz'], axis=1)
    d['grp'] = d.lesion.str.replace(r'random400_s\d', 'random', regex=True)
    d.to_csv(out / 'fig5_data.csv', index=False)
    settings = list(base.index)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=False)
    for a, col, ttl in [(ax[0], 'HT_rel', 'Serotonergic signal / intact'), (ax[1], 'MN9_rel', 'Feeding (MN9) / intact')]:
        for j, (g, c) in enumerate([('PRW', C['tissue']), ('CB0246', C['neuron']), ('random', C['random'])]):
            sub = d[d.grp == g]
            xs = [settings.index(s) + (j - 1) * 0.22 for s in sub.setting]
            a.scatter(xs, sub[col], color=c, s=26, label=g if a is ax[0] else None, edgecolor='none')
        a.set_xticks(range(len(settings)), [s.replace('_w', '\nW=') for s in settings])
        a.axhline(1, ls=':', color='0.6'); a.set(title=ttl, ylim=(-0.05, 1.3))
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'fig5_replication.png', dpi=200); plt.close(fig)


def fig_spheres(out):
    """Fig 6: round infarct-like strokes at PRW + random null."""
    names = ['INTACT', 'PRW_r45', 'PRW_r45_pen', 'PRW_r60', 'PRW_r60_pen', 'RAND_PRW_r45_s0', 'RAND_PRW_r60_s0']
    rows = []
    for n in names:
        p = R / 'a2' / 'rates' / f'{n}.csv'
        if p.exists():
            r = rd(p)
            rows.append((n, r.get(MN9, 0), ht(r), r.get(720575940632602532, 0)))
    nul = pd.read_csv(R / 'replication' / 'round_stroke_null.csv')
    for _, x in nul.iterrows():
        rows.append((f'null_s{int(x.seed)}', x.MN9_Hz, x[['5HT_0', '5HT_1', '5HT_2']].mean(), x.CB0246_Hz))
    d = pd.DataFrame(rows, columns=['cond', 'MN9', 'HT', 'CB0246'])
    d.to_csv(out / 'fig6_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    isnull = d.cond.str.startswith('null')
    ax.scatter(d[isnull].MN9, d[isnull].HT, color=C['random'], s=26, label='random, size-matched to r60+penumbra (10)')
    for _, x in d[~isnull].iterrows():
        col = C['random'] if x.cond.startswith('RAND') else (C['neuron'] if 'pen' in x.cond else C['tissue'])
        ax.scatter(x.MN9, x.HT, color=col, s=34, edgecolor='k', linewidth=0.4)
        ax.annotate(x.cond.replace('PRW_', ''), (x.MN9, x.HT), xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.set(xlabel='MN9 (Hz)', ylabel='Serotonergic cells, mean (Hz)', title='Round strokes centred on PRW')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'fig6_round_strokes.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_atlas, fig_bottleneck, fig_replication, fig_spheres):
        f(out)
        print('ok', f.__name__)
