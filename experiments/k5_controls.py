"""K5: cutoff sweep, 200 matched round strokes, pairwise chain knockouts.

All with sugar on both sides (K3 input), v783, default W_syn, 100 Hz. Readouts: 6 serotonin
cells, pump MNs, positioning MNs (K4 groups). Hit rule fixed in advance (round strokes):
serotonin < 25% of intact AND positioning > 85% of intact.

1. Cutoff sweep: PRW neuron lesion at membership cutoffs 0.3 to 0.7, 10 trials.
2. Round strokes: 200 lesions of the 402 neurons nearest a random central-brain neuron
   (contiguous, size matched to the PRW lesion), 5 trials; records which chain cells are hit.
3. Pairwise knockouts: 6 chain cells (CB0883, CB0559, CB0246; L and R), all singles and
   pairs, 10 trials.
Stimulated GRNs and readout cells are never lesioned.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, default_params, V783  # noqa: E402
from ffsim.lesion import load_neuropil, load_positions  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR  # noqa: E402
from experiments.k2_bilateral import SEL, CB0246_L, CB0246_R, CB0559_L, CB0559_R  # noqa: E402
from experiments.k3_bilateral_sugar import right_sugar, CB0883_L, CB0883_R  # noqa: E402
from experiments.k4_robust_mirror import groups, run_set  # noqa: E402

OUT = ROOT / 'results' / 'k5'
CHAIN = {'CB0246_L': CB0246_L, 'CB0246_R': CB0246_R, 'CB0559_L': CB0559_L,
         'CB0559_R': CB0559_R, 'CB0883_L': CB0883_L, 'CB0883_R': CB0883_R}
N_STROKES, STROKE_SIZE = 200, 402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    sugar = [f for f in NEU_SUGAR if f in ids] + right_sugar(ids)
    pump, pos = groups()
    spare = set(sugar) | set(SEL.values()) | set(pump) | set(pos)
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    t_run_s = float(p['t_run'] / second)
    model = Model(sugar, p, data=V783)

    print('=== 1. cutoff sweep ===', flush=True)
    npt = load_neuropil()
    prw = npt[npt.neuropil == 'PRW']
    conds, sizes = {'intact': ([], None)}, []
    for c in [0.3, 0.4, 0.5, 0.6, 0.7]:
        les = [f for f in prw.flywire_id[prw.frac >= c] if f in ids and f not in spare]
        conds[f'PRW_{int(c * 100)}'] = (les, None)
        sizes.append({'cutoff': c, 'n': len(les), 'CB0246_L': CB0246_L in les, 'CB0246_R': CB0246_R in les})
    print(pd.DataFrame(sizes).to_string(index=False), flush=True)
    pd.DataFrame(sizes).to_csv(OUT / 'cutoff_sizes.csv', index=False)
    run_set(model, conds, 10, OUT / 'cutoff', t_run_s, pump, pos)

    print('=== 3. pairwise chain knockouts ===', flush=True)
    conds = {'intact': ([], None)}
    for k in (1, 2):
        for combo in combinations(CHAIN, k):
            conds['+'.join(combo)] = ([CHAIN[n] for n in combo], None)
    run_set(model, conds, 10, OUT / 'pairs', t_run_s, pump, pos)

    print('=== 2. round strokes ===', flush=True)
    posn = load_positions()
    posn = posn[posn.index.isin(ids)]
    ann = pd.read_csv(ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv', sep='\t',
                      usecols=['root_id', 'super_class'], low_memory=False).drop_duplicates('root_id')
    centers_pool = sorted(set(ann.root_id[ann.super_class == 'central']) & set(posn.index))
    cand = posn[~posn.index.isin(spare)]
    xyz = cand[['x', 'y', 'z']].to_numpy()
    rng = np.random.default_rng(2026)
    centers = rng.choice(centers_pool, size=N_STROKES, replace=False)
    conds, meta = {'intact': ([], None)}, []
    for s, cid in enumerate(centers):
        d = np.linalg.norm(xyz - posn.loc[cid, ['x', 'y', 'z']].to_numpy(float), axis=1)
        les = list(cand.index[np.argsort(d)[:STROKE_SIZE]])
        conds[f'stroke{s:03d}'] = (les, None)
        meta.append({'cond': f'stroke{s:03d}', 'center': cid, 'radius_um': float(np.sort(d)[STROKE_SIZE - 1]),
                     **{f'hit_{k}': v in les for k, v in CHAIN.items()}})
    pd.DataFrame(meta).to_csv(OUT / 'strokes_meta.csv', index=False)
    res = run_set(model, conds, 5, OUT / 'strokes', t_run_s, pump, pos)
    base = res[res.cond == 'intact'].iloc[0]
    r = res[res.cond != 'intact'].merge(pd.DataFrame(meta), on='cond')
    r['sel_rel'] = r.sel6_mean / base.sel6_mean
    r['pos_rel'] = r.position_mean / base.position_mean
    r['pump_rel'] = r.pump_mean / base.pump_mean
    r['hit'] = (r.sel_rel < 0.25) & (r.pos_rel > 0.85)
    r['hits_chain'] = r[[f'hit_{k}' for k in CHAIN]].any(axis=1)
    r.to_csv(OUT / 'strokes_scored.csv', index=False)
    print(f'hits: {r.hit.sum()} / {len(r)}', flush=True)
    print(pd.crosstab(r.hits_chain, r.hit, margins=True).to_string(), flush=True)


if __name__ == '__main__':
    main()
