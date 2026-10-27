"""Q5: does wiring + one intact run predict the PRW knockout screen?

No simulation. Scores each of the 402 PRW neurons by
  static:      synapse-weighted sugar->5-HT paths through it (as p1_static_paths.py)
  active:      its intact sugar-evoked rate alone
  act_direct:  rate x signed synapses onto the 3 5-HT cells
  act_paths:   paths through it to the 5-HT cells (<= 3 hops), each edge weighted by
               synapses x presynaptic intact rate x sign (only active neurons carry signal)
and compares each with the knockout effect (results_summary/k1_screen_ranked.csv).

usage: python experiments/q5_activity_map.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e0_baseline import NEU_SUGAR  # noqa: E402
from experiments.p1_static_paths import load, through_scores, HT, KO_HITS  # noqa: E402

S = ROOT / 'results_summary'


def main():
    ids, A, exc = load()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    src = pos.reindex(NEU_SUGAR).dropna().astype(int).to_numpy()
    dst = pos[HT].to_numpy()
    r = pd.read_csv(ROOT / 'results/a1/rates/INTACT.csv', index_col=0).r
    rate = np.zeros(len(ids))
    rate[pos.reindex(r.index).dropna().astype(int).to_numpy()] = r.reindex(
        pos.reindex(r.index).dropna().index).to_numpy()
    sign = np.where(exc, 1.0, -1.0)

    ko = pd.read_csv(S / 'k1_screen_ranked.csv')
    ko['flywire_id'] = ko.cond.astype('int64')
    prw = pos[ko.flywire_id].to_numpy()
    effect = 1 - ko['5HT_rel'].to_numpy()  # >0 = 5-HT drops when silenced
    hit = np.isin(ids[prw], list(KO_HITS))

    static, _, _ = through_scores(A, src, dst, 3, exc)
    Ar = (sp.diags(rate * sign) @ A).tocsr()
    act_paths, _, _ = through_scores(Ar, src, dst, 3, rate > 0)
    act_direct = rate * sign * np.asarray(A[:, dst].sum(axis=1)).ravel()
    scores = {'static_paths': static, 'active_rate': rate, 'act_direct': act_direct, 'act_paths': act_paths}

    rows = []
    for k, s in scores.items():
        x = s[prw]
        order = np.argsort(-x, kind='stable')
        hit_ranks = sorted(int(np.where(order == i)[0][0]) + 1 for i in np.flatnonzero(hit))
        rows.append({'score': k, 'spearman_vs_effect': spearmanr(x, effect).correlation,
                     'hit_ranks_of_402': hit_ranks, 'hits_in_top5': sum(h <= 5 for h in hit_ranks),
                     'hits_in_top10': sum(h <= 10 for h in hit_ranks),
                     'n_nonzero': int((x != 0).sum())})
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(f'PRW neurons active under sugar (intact): {(rate[prw] > 0).sum()} of {len(prw)}')
    print(res.to_string(index=False))
    res.to_csv(S / 'q5_activity_map.csv', index=False)

    top = np.argsort(-act_paths[prw])[:8]
    print('\ntop PRW neurons by act_paths:')
    print(pd.DataFrame({'flywire_id': ids[prw[top]], 'rate_Hz': rate[prw[top]],
                        'act_paths': act_paths[prw[top]], 'KO_5HT_rel': ko['5HT_rel'].to_numpy()[top],
                        'hit': hit[top]}).to_string(index=False))


if __name__ == '__main__':
    main()
