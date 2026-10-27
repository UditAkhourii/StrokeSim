"""Figure 7 for D1: serotonin vs swallowing, one point per condition.
Reads results_summary/d1_summary.csv only. Shaded zones are the pre-registered thresholds:
  serotonin lost, swallowing intact: 5-HT < 0.25 and pump > 0.8
  swallowing lost, serotonin intact: pump < 0.6 and 5-HT > 0.75

usage: python experiments/d1_figure.py [OUT_DIR]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'results_summary'
X0, X1, Y0, Y1 = -0.05, 1.25, -0.12, 1.2
LABEL = {
    'C0_intact': ('C0  intact', (12, 10)),
    'C1_CB0246L_ko': ('C1  CB0246-L silenced', (-20, -22)),
    'C2_cut_L_to_SEL': ('C2  cut L→serotonin cells', (-170, 14)),
    'C3_cut_L_except_SEL': ('C3  cut all L outputs\nexcept serotonin cells', (-60, 18)),
    'C4_cut_LR_to_SEL': ('C4  cut L+R→serotonin cells', (-150, -22)),
    'C5_SEL6_ko': ('C5  serotonin-cell output removed', (-200, 12)),
    'C6_module5_ko': ('C6  5 module neurons silenced', (8, -22)),
}
KEY = {'C2_cut_L_to_SEL', 'C4_cut_LR_to_SEL'}

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def main(out):
    d = pd.read_csv(S / 'd1_summary.csv')
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.fill_between([0.8, X1], Y0, 0.25, color='#2a6f97', alpha=0.12, lw=0)
    ax.text(1.02, 0.31, 'serotonin lost,\nswallowing intact', color='#2a6f97', ha='center',
            fontsize=9, style='italic')
    ax.fill_between([X0, 0.6], 0.75, Y1, color='#a44a3f', alpha=0.12, lw=0)
    ax.text(0.275, 0.95, 'swallowing lost,\nserotonin intact\n(no condition here)', color='#a44a3f',
            ha='center', fontsize=9, style='italic')
    ax.axhline(1, ls=':', color='0.75', lw=0.8)
    ax.axvline(1, ls=':', color='0.75', lw=0.8)
    for _, r in d.iterrows():
        key = r.cond in KEY
        c = '#2a6f97' if key else ('0.25' if r.cond == 'C0_intact' else '0.55')
        ax.errorbar(r.pump_sum_rel, r.sel6_mean_rel, xerr=r.pump_sum_rel_sd, yerr=r.sel6_mean_rel_sd,
                    fmt='o', ms=8 if key else 6, color=c, mec='white', capsize=3, lw=1.2, zorder=3)
        txt, off = LABEL[r.cond]
        ax.annotate(txt, (r.pump_sum_rel, r.sel6_mean_rel), xytext=off, textcoords='offset points',
                    fontsize=8.5, color=c)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_xlabel('Swallowing: summed pump motor-neuron rate / intact')
    ax.set_ylabel('Serotonin: mean rate of the 6 SEL cells / intact')
    ax.set_title("Cutting CB0246's synapses onto the serotonin cells removes the serotonin\n"
                 'signal with swallowing intact; the reverse has no match', fontsize=10.5, loc='left')
    fig.text(0.01, 0.005, 'Shaded: pre-registered thresholds. Error bars: trial SD / intact mean (20 trials).',
             fontsize=7.5, color='0.4')
    fig.tight_layout()
    fig.savefig(out / 'fig7_dissociation.png', dpi=200)


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    main(out)
