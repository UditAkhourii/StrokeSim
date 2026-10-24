"""E1b option 4: odour stand-in probe. usage: e1b_odor_probe.py {orn|pn} {odor|glomerulus} RATE_HZ (200 ms, 1 trial)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from brian2 import Hz, ms
from ffsim.sim import run, default_params, V783
from ffsim.readouts import load_annotations, orn_ids, ODORS
a = load_annotations(); ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
lbl = a.set_index('root_id'); lbl = lbl.cell_class.fillna(lbl.super_class)
mode, odor, rate = sys.argv[1], sys.argv[2], float(sys.argv[3])
glom = ODORS[odor] if odor in ODORS else [odor]
if mode == 'orn':
    exc = orn_ids(glom, a, ids)
else:
    pn = a[(a.cell_class == 'ALPN') & (a.cell_sub_class == 'uniglomerular')]
    exc = [f for f, t in zip(pn.root_id, pn.cell_type.astype(str)) if t.split('_')[0] in glom and f in ids]
p = dict(default_params); p['r_poi'] = rate * Hz; p['t_run'] = 200 * ms
df = run(exc, p, data=V783, n_run=1, verbose=False)
n = df.groupby('flywire_id').size(); c = lbl.reindex(n.index).value_counts(); tot = lbl.value_counts()
f = lambda k: f"{c.get(k, 0)}/{tot[k]}"
print(f"{mode} {odor:8s} {rate:>4.0f}Hz in={len(exc):3d} active={len(n):5d}  PN {f('ALPN')}  LN {f('ALLN')}  KC {f('Kenyon_Cell')}  MBON {f('MBON')}  DAN {f('DAN')}", flush=True)
