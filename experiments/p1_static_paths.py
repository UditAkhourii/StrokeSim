"""P1: does the static wiring map already predict CB0246?

No simulation. Scores every neuron by the synapse-weighted paths from the sugar GRNs to the
three serotonergic SEZ cells that pass through it, for paths of up to --max-hops synapses.
Path weight = product of edge weights; edges are either raw synapse counts or the fraction of
the postsynaptic cell's input (input-normalised, as in effective-connectivity analyses).
'exc' variants use only paths whose intermediate neurons are excitatory.

Rankings are compared with (a) the direct-input ranking onto the 5-HT cells and (b) the
single-neuron knockout screen over the 402 PRW neurons (results_summary/k1_screen_ranked.csv).

usage: python experiments/p1_static_paths.py [--max-hops 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffsim.sim import V783  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR  # noqa: E402

HT = [720575940636895406, 720575940631179831, 720575940628676457]
CB0246 = 720575940632602532
KO_HITS = {720575940632602532: 'CB0246', 720575940625102692: 'CB0883', 720575940620780272: 'CB0559'}
S = ROOT / 'results_summary'


def load():
    comp = pd.read_csv(V783['path_comp'], index_col=0)
    ids = comp.index.to_numpy()
    con = pd.read_parquet(V783['path_con'], columns=['Presynaptic_Index', 'Postsynaptic_Index',
                                                     'Connectivity', 'Excitatory'])
    n = len(ids)
    A = sp.csr_matrix((con.Connectivity.to_numpy(float),
                       (con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy())), shape=(n, n))
    exc = np.zeros(n, bool)
    exc[con.Presynaptic_Index[con.Excitatory == 1].unique()] = True
    return ids, A, exc


def through_scores(A, src, dst, max_hops, mask=None):
    """Weight of all src->dst paths of length <= max_hops passing through each node.

    f_in[h][j]: summed weight of length-h paths src->j; f_out[h][j]: length-h paths j->dst.
    Paths through j with h_in + h_out <= max_hops contribute f_in[h_in][j] * f_out[h_out][j].
    mask: nodes allowed as intermediates (sources/targets always allowed).
    """
    n = A.shape[0]
    M = A if mask is None else A.multiply(mask[:, None]).tocsr()  # mask applies to presynaptic node
    s = np.zeros(n); s[src] = 1
    t = np.zeros(n); t[dst] = 1
    f_in, f_out = [None], [None]
    v = s.copy()
    vs = A.T @ s  # first hop leaves a source, always allowed
    f_in.append(vs)
    v = vs
    for _ in range(max_hops - 2):
        v = M.T @ v
        f_in.append(v)
    w = A @ t  # last hop into a target
    f_out.append(w)
    for _ in range(max_hops - 2):
        w = M @ w
        f_out.append(w)
    score = np.zeros(n)
    for hi in range(1, len(f_in)):
        for ho in range(1, len(f_out)):
            if hi + ho <= max_hops:
                score += f_in[hi] * f_out[ho]
    if mask is not None:
        score = score * mask
    return score, f_in, f_out


def rank_of(score, idx, pool):
    """1-based rank of idx within pool (descending score); NaN if idx has no path."""
    if score[idx] <= 0:
        return np.nan
    return int((score[pool] > score[idx]).sum()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-hops', type=int, default=3)
    args = ap.parse_args()

    ids, A, exc = load()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    src = pos.reindex(NEU_SUGAR).dropna().astype(int).to_numpy()
    dst = pos[HT].to_numpy()
    cb = pos[CB0246]
    print(f'{len(ids)} neurons, {A.nnz} edges, {len(src)} sugar GRNs')

    insum = np.asarray(A.sum(axis=0)).ravel()
    An = A @ sp.diags(1 / np.maximum(insum, 1))  # fraction of postsynaptic input
    An = An.tocsr()

    ko = pd.read_csv(S / 'k1_screen_ranked.csv')
    ko['flywire_id'] = ko.cond.astype('int64')
    prw = pos.reindex(ko.flywire_id).to_numpy()
    direct_pool = np.unique(A[:, dst].nonzero()[0])  # all presynaptic partners of the 5-HT cells

    rows, per_neuron = [], {}
    for wname, W in [('synapses', A), ('input_frac', An)]:
        for mname, mask in [('all', None), ('exc', exc)]:
            for h in range(2, args.max_hops + 1):
                sc, _, _ = through_scores(W, src, dst, h, mask)
                key = f'{wname}_{mname}_h{h}'
                per_neuron[key] = sc
                rho = spearmanr(sc[prw], -ko['5HT_rel'].to_numpy()).correlation
                prw_order = prw[np.argsort(-sc[prw], kind='stable')]
                top3 = set(ids[prw_order[:3]]) & set(KO_HITS)
                rows.append({
                    'weights': wname, 'paths': mname, 'max_hops': h,
                    'CB0246_rank_all_neurons': rank_of(sc, cb, np.flatnonzero(sc > 0)),
                    'n_scored': int((sc > 0).sum()),
                    'CB0246_rank_5HT_inputs': rank_of(sc, cb, direct_pool),
                    'n_5HT_inputs': len(direct_pool),
                    'CB0246_rank_PRW': rank_of(sc, cb, prw),
                    'CB0246_share_of_paths': sc[cb] / sc[direct_pool].sum() if h == 2 else np.nan,
                    'KO_hits_in_PRW_top3': len(top3),
                    'spearman_vs_KO': rho,
                })
    # direct-synapse baseline (what "static map" meant before)
    d = np.asarray(A[:, dst].sum(axis=1)).ravel()
    rows.append({'weights': 'synapses', 'paths': 'direct', 'max_hops': 1,
                 'CB0246_rank_5HT_inputs': rank_of(d, cb, direct_pool), 'n_5HT_inputs': len(direct_pool),
                 'CB0246_rank_PRW': rank_of(d, cb, prw),
                 'KO_hits_in_PRW_top3': len(set(ids[prw[np.argsort(-d[prw], kind='stable')][:3]]) & set(KO_HITS)),
                 'spearman_vs_KO': spearmanr(d[prw], -ko['5HT_rel'].to_numpy()).correlation})
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(res.to_string(index=False))
    res.to_csv(S / 'p1_static_path_ranks.csv', index=False)

    # top 10 5-HT inputs on the main measure, with knockout effect where available
    key = f'input_frac_all_h{args.max_hops}'
    sc = per_neuron[key]
    top = direct_pool[np.argsort(-sc[direct_pool])][:10]
    ko_rel = ko.set_index('flywire_id')['5HT_rel']
    t = pd.DataFrame({'flywire_id': ids[top], 'path_score': sc[top],
                      'direct_syn_onto_5HT': d[top],
                      'KO_5HT_rel': [ko_rel.get(i, np.nan) for i in ids[top]],
                      'name': [KO_HITS.get(i, '') for i in ids[top]]})
    print(f'\ntop 5-HT inputs by {key}:\n', t.to_string(index=False))
    t.to_csv(S / 'p1_top_inputs.csv', index=False)


if __name__ == '__main__':
    main()
