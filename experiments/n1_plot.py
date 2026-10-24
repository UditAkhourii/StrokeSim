"""Fig 3: W_syn x stimulus phase diagram from N1."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / 'results' / 'n1' / 'summary.csv').sort_values('w_syn_mV')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'n1' / 'fig3_wsyn.png'

col = {'sugar': '#2a6f97', 'bitter': '#a44a3f'}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for stim, g in d.groupby('stim'):
    kw = dict(marker='o', color=col[stim], label=stim)
    ax[0].plot(g.w_syn_mV, g.PAM_frac * 100, **kw)
    ax[1].plot(g.w_syn_mV, g.active_total, **kw)
    ax[2].plot(g.w_syn_mV, g.KC_frac * 100, **kw)
    ax[3].plot(g.w_syn_mV, g.MN9_Hz, **kw)
ax[1].set_yscale('log')
for a, (t, yl) in zip(ax, [('PAM recruited', '% PAM >1 Hz'), ('Network size', '# active neurons'),
                            ('Kenyon cells', '% KC >1 Hz'), ('Feeding output', 'MN9 (Hz)')]):
    a.axvspan(0.48, 0.92, color='0.9', zorder=0)
    a.axvline(0.275, ls=':', color='0.4')
    a.set(title=t, xlabel='W_syn (mV)', ylabel=yl)
    a.spines[['top', 'right']].set_visible(False)
ax[0].legend(frameon=False)
ax[0].text(0.5, 50, 'ignited', fontsize=9, color='0.4')
fig.suptitle('N1: raising global synaptic gain recruits PAM only via valence-blind ignition (v783, 100 Hz GRN input, 10 trials)')
fig.tight_layout()
fig.savefig(out, dpi=150)
print(out)
