"""Plot E0 dose-response: sugar GRN stimulation rate vs MN9 rate and network size."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'results' / 'e0'
df = pd.read_csv(OUT / 'summary.csv').sort_values('r_poi_Hz')

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].errorbar(df.r_poi_Hz, df.MN9_Hz, yerr=df.MN9_sd, marker='o', capsize=3, color='#2a6f97')
ax[0].set(xlabel='Sugar GRN stimulation (Hz)', ylabel='MN9 rate (Hz)', title='Sugar GRN → MN9')
ax[1].plot(df.r_poi_Hz, df.active_neurons, marker='o', color='#a44a3f')
ax[1].set(xlabel='Sugar GRN stimulation (Hz)', ylabel='# neurons active', title='Recruited network size')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.suptitle('E0 baseline — FlyWire v630, 21 right sugar GRNs, 30 trials × 1 s')
fig.tight_layout()
fig.savefig(OUT / 'e0_dose_response.png', dpi=150)
print(OUT / 'e0_dose_response.png')
