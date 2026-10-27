"""README headline figure: serotonergic signal after targeted vs random strokes.
Reads results_summary/ only (no simulation, no full results needed).

usage: python experiments/readme_figures.py OUT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'results_summary'
HT = ['5HT_0', '5HT_1', '5HT_2']
C = {'random': '0.6', 'PRW': '#2a6f97', 'CB0246': '#a44a3f'}

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def null_figure(out):
    """5-HT signal / intact for PRW stroke, CB0246 alone, and 22 random 400-neuron strokes."""
    rows = []
    # main setting: v783, W_syn 0.275
    a1 = pd.read_csv(S / 'a1_intact_prw_5ht.csv').set_index('cond')
    base = a1.loc['INTACT', HT].mean()
    rows.append(('PRW', 'v783 W=0.275', a1.loc['PRW', HT].mean() / base))
    a3 = pd.read_csv(S / 'a3_summary.csv').set_index('silenced')
    rows.append(('CB0246', 'v783 W=0.275', a3.loc['CB0246', HT].mean() / base))
    for _, x in pd.read_csv(S / 'a1b_random400_null.csv').iterrows():
        rows.append(('random', 'v783 W=0.275', x[HT].mean() / base))
    # replication settings
    rep = pd.read_csv(S / 'replication_robustness.csv')
    for setting, g in rep.groupby('setting'):
        b = g[g.lesion == 'intact']['5HT_mean_Hz'].iloc[0]
        for _, x in g[g.lesion != 'intact'].iterrows():
            kind = 'random' if x.lesion.startswith('random') else x.lesion
            rows.append((kind, setting.replace('_w', ' W='), x['5HT_mean_Hz'] / b))
    d = pd.DataFrame(rows, columns=['kind', 'setting', 'ratio'])
    d.to_csv(out / 'null_data.csv', index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for i, k in enumerate(['random', 'PRW', 'CB0246']):
        g = d[d.kind == k]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g.ratio, s=34, color=C[k],
                   edgecolor='white', linewidth=0.6, zorder=3)
        ax.text(i, 1.45, f'n = {len(g)}', ha='center', fontsize=9, color='0.35')
    ax.axhline(1, ls=':', color='0.5')
    ax.set_xticks([0, 1, 2], ['22 random strokes\n(400 neurons each)', 'PRW stroke\n(402 neurons)',
                              'CB0246 alone\n(1 neuron)'])
    ax.set_ylim(-0.08, 1.55)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel('Serotonergic signal\n(3 cells, / intact)')
    ax.set_title('Only targeted lesions silence the serotonergic cells', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'fig0_random_vs_targeted.png', dpi=200)
    print(d.groupby('kind').ratio.describe())


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    null_figure(out)
