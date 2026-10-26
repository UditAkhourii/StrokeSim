"""Build Shiu-format model inputs (completeness csv + connectivity parquet) from the
Janelia male CNS v1.0 flat connectome (data/malecns/).

Neurons: bodies with status == 'Traced'. Sign: -1 if consensus_nt (fallback predicted_nt)
is gaba or glutamate, else +1 (as Shiu et al.). Edges: weight >= MIN_SYN, chosen so the
edge count is close to FlyWire v783 in the Shiu model (15.1M edges, 8 GB laptop limit).
Usage: build_malecns_model.py count | build MIN_SYN
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

D = Path(__file__).resolve().parents[1] / 'data' / 'malecns'


def traced():
    a = pd.read_feather(D / 'ann.feather', columns=['bodyId', 'status'])
    return np.sort(a.bodyId[a.status == 'Traced'].to_numpy())


def batches(keep):
    keep = pa.array(keep)
    for b in pf.read_table(D / 'weights.feather', memory_map=True).to_batches(max_chunksize=5_000_000):
        m = pc.and_(pc.is_in(b['body_pre'], value_set=keep), pc.is_in(b['body_post'], value_set=keep))
        yield b.filter(m)


def main():
    keep = traced()
    if sys.argv[1] == 'count':
        cnt = np.zeros(11, dtype=np.int64)
        for b in batches(keep):
            w = b['weight'].to_numpy()
            for t in range(1, 11):
                cnt[t] += (w >= t).sum()
        print(f'{len(keep)} traced bodies')
        for t in range(1, 11):
            print(f'weight >= {t}: {cnt[t]:,} edges')
        return
    t = int(sys.argv[2])
    parts = [b.filter(pc.greater_equal(b['weight'], t)).to_pandas() for b in batches(keep)]
    e = pd.concat(parts, ignore_index=True)
    nt = pd.read_feather(D / 'nt.feather', columns=['body', 'consensus_nt', 'predicted_nt']).set_index('body')
    tx = nt.consensus_nt.fillna(nt.predicted_nt).reindex(keep)
    sign = pd.Series(np.where(tx.isin(['gaba', 'glutamate']), -1, 1), index=keep)
    idx = pd.Series(np.arange(len(keep)), index=keep)
    out = pd.DataFrame({'Presynaptic_ID': e.body_pre.values, 'Postsynaptic_ID': e.body_post.values,
                        'Presynaptic_Index': idx[e.body_pre].values, 'Postsynaptic_Index': idx[e.body_post].values,
                        'Connectivity': e.weight.values, 'Excitatory': sign[e.body_pre].values})
    out['Excitatory x Connectivity'] = out.Connectivity * out.Excitatory
    out.to_parquet(D / f'Connectivity_malecns_min{t}.parquet')
    pd.DataFrame({'Completed': True}, index=pd.Index(keep, name=None)).to_csv(D / 'Completeness_malecns.csv')
    print(f'{len(keep)} neurons, {len(out):,} edges, frac inhibitory {(out.Excitatory < 0).mean():.2f}')


if __name__ == '__main__':
    main()
