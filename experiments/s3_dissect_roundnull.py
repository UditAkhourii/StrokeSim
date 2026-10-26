"""S3: why does round-stroke null seed 3 nearly abolish the 5-HT response?

Structural analysis only (no simulation). Rebuilds the 10 round-stroke null lesions
exactly as a5r_replication.py does (935 silenced + 3711 at 50% outgoing weight,
pool.sample(random_state=300 + s)), then for every seed reports:
  - whether CB0246 and the readout cells are silenced, halved, or untouched,
  - the fraction of each target's excitatory and inhibitory synaptic input removed
    (silenced = 100% of that presynaptic neuron's weight, penumbra = 50%),
for targets CB0246, the three 5-HT cells and MN9. Also lists the strongest inputs to
the 5-HT cells and CB0246 and which seed-3 lesion class each falls in.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

SHIU = ROOT / 'external' / 'Drosophila_brain_model'
OUT = ROOT / 'results' / 's3'
HT = [720575940636895406, 720575940631179831, 720575940628676457]
CB0246 = 720575940632602532
TARGETS = {'CB0246': CB0246, '5HT_0': HT[0], '5HT_1': HT[1], '5HT_2': HT[2], 'MN9': ID_MN9}


def round_null(seed, ids):
    sugar = [f for f in NEU_SUGAR if f in ids]
    pool = pd.Series(sorted(ids - set(sugar) - set(HT) - {ID_MN9}))
    pick = pool.sample(935 + 3711, random_state=300 + seed)
    return set(pick[:935]), set(pick[935:])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(SHIU / 'Completeness_783.csv', index_col=0).index)
    con = pd.read_parquet(SHIU / 'Connectivity_783.parquet',
                          columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Excitatory x Connectivity'])
    con = con[con.Postsynaptic_ID.isin(TARGETS.values())].rename(
        columns={'Presynaptic_ID': 'pre', 'Postsynaptic_ID': 'post', 'Excitatory x Connectivity': 'w'})
    name = {v: k for k, v in TARGETS.items()}
    con['target'] = con.post.map(name)

    rows = []
    for s in range(10):
        slnc, pen = round_null(s, ids)
        loss = con.pre.map(lambda f: 1.0 if f in slnc else 0.5 if f in pen else 0.0)
        row = {'seed': s, 'CB0246_hit': 'silenced' if CB0246 in slnc else 'penumbra' if CB0246 in pen else '-'}
        for t in TARGETS:
            c = con.target == t
            exc, inh = c & (con.w > 0), c & (con.w < 0)
            row[f'{t}_exc_lost'] = (con.w[exc] * loss[exc]).sum() / con.w[exc].sum()
            row[f'{t}_inh_lost'] = (con.w[inh] * loss[inh]).sum() / con.w[inh].sum() if inh.any() else 0.0
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'input_loss_by_seed.csv', index=False)
    pd.set_option('display.width', 250)
    print(d.round(3).to_string(index=False))

    # strongest inputs to CB0246 and the 5-HT cells, labelled with their seed-3 fate
    slnc, pen = round_null(3, ids)
    ann = pd.read_csv(ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv', sep='\t',
                      usecols=['root_id', 'cell_type', 'hemibrain_type', 'super_class', 'top_nt'],
                      low_memory=False).drop_duplicates('root_id').set_index('root_id')
    top = con[con.target != 'MN9'].copy()
    top['seed3'] = top.pre.map(lambda f: 'silenced' if f in slnc else 'penumbra' if f in pen else '-')
    top = top.join(ann, on='pre')
    top['rank'] = top.groupby('target').w.transform(lambda x: x.abs().rank(ascending=False))
    top = top[top['rank'] <= 10].sort_values(['target', 'rank'])
    top.to_csv(OUT / 'top_inputs_seed3.csv', index=False)
    print(top[['target', 'pre', 'w', 'seed3', 'cell_type', 'hemibrain_type', 'top_nt']].to_string(index=False))


if __name__ == '__main__':
    main()
