"""Stroke analysis on E2 per-neuron rates: which lesions cut reward-side readouts but spare feeding?

Readouts (rates, Hz): MN9 (motor), SEZ-MA (9 sugar-specific SEZ neurons), SEZ-5HT
(the 3 immuno-serotonergic members), CB0233 (fox->FDA valence route, Christie 2025).
Indices are lesion / intact. A lesion is a hit if a readout falls outside the
range of the random size-matched controls.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / 'results' / 'e2'
MN9 = 720575940660219265
SEZ_MA = list(pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id)
SEZ_5HT = [720575940636895406, 720575940631179831, 720575940628676457]  # CB0262, CB0892 x2
CB0233 = [720575940617648453, 720575940655011489]


def readouts(r):
    return {'MN9': r.get(MN9, 0.0),
            'SEZ_MA': r.reindex(SEZ_MA).fillna(0).mean(),
            'SEZ_5HT': r.reindex(SEZ_5HT).fillna(0).mean(),
            'CB0233': r.reindex(CB0233).fillna(0).max(),
            'active': int((r > 1).sum())}


def main(out_dir=None):
    rows = []
    for p in sorted((E2 / 'rates').glob('*.csv')):
        r = pd.read_csv(p, index_col=0).r
        rows.append({'cond': p.stem.replace('__', ':'), **readouts(r)})
    d = pd.DataFrame(rows)
    base = d[d.cond.str.lower().str.startswith('intact')].iloc[0]
    for k in ['MN9', 'SEZ_MA', 'SEZ_5HT', 'CB0233']:
        d[f'{k}_rel'] = d[k] / base[k] if base[k] else float('nan')
    d['kind'] = d.cond.str.split(':').str[0]
    # summary rows carry PAM Vm; merge if present
    conds = E2 / 'conds'
    if conds.exists():
        s = pd.concat([pd.read_csv(p) for p in conds.glob('*.csv')], ignore_index=True)
        pam = [c for c in s.columns if 'PAM' in c]
        key = [c for c in s.columns if c in ('cond', 'condition', 'name')]
        if pam and key:
            d = d.merge(s[[key[0]] + pam].rename(columns={key[0]: 'cond'}), on='cond', how='left')
    rnd = d[d.kind == 'random']
    lo = {k: rnd[f'{k}_rel'].min() for k in ['MN9', 'SEZ_MA', 'SEZ_5HT', 'CB0233']} if len(rnd) else {}
    for k, v in lo.items():
        d[f'{k}_hit'] = (d.kind != 'random') & (d[f'{k}_rel'] < min(v, 0.8))
    d['reward_vs_motor'] = d[['SEZ_5HT_rel', 'CB0233_rel']].min(axis=1) - d.MN9_rel
    d['dissociation'] = d.MN9_rel - d[['SEZ_MA_rel', 'SEZ_5HT_rel', 'CB0233_rel']].min(axis=1)
    out = Path(out_dir) if out_dir else E2
    d.sort_values('dissociation', ascending=False).to_csv(out / 'stroke_atlas.csv', index=False)
    show = ['cond', 'MN9_rel', 'SEZ_MA_rel', 'SEZ_5HT_rel', 'CB0233_rel', 'active', 'dissociation']
    print('random-control floor (min rel):', {k: round(v, 2) for k, v in lo.items()})
    print('\nTop reward-side-loss / motor-spared lesions:')
    print(d[d.kind != 'random'].sort_values('dissociation', ascending=False)[show].head(15).round(2).to_string(index=False))
    print('\nTop motor-loss lesions:')
    print(d[d.kind != 'random'].sort_values('MN9_rel')[show].head(8).round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, y in zip(ax, ['SEZ_5HT_rel', 'CB0233_rel']):
        for kind, c in [('tissue', '#2a6f97'), ('neuron', '#a44a3f'), ('random', '0.6')]:
            g = d[d.kind == kind]
            a.scatter(g.MN9_rel, g[y], s=18, color=c, label=kind, alpha=0.8)
        a.plot([0, 1.3], [0, 1.3], ls=':', color='0.5')
        a.set(xlabel='Feeding (MN9 / intact)', ylabel=f'{y.replace("_rel", "")} / intact',
              title=f'Stroke atlas: motor vs {y.replace("_rel", "")}')
        for _, rr in d[(d.kind != 'random') & (d.dissociation > 0.3)].iterrows():
            a.annotate(rr.cond.split(':')[-1], (rr.MN9_rel, rr[y]), fontsize=7)
        a.spines[['top', 'right']].set_visible(False)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'fig5_stroke_atlas.png', dpi=150)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
