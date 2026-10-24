"""E2: single-neuropil lesion atlas (v783, sugar GRNs at 100 Hz).

Lesion types
  tissue:<NP>   every synapse located in neuropil NP removed (connection weight
                scaled by the fraction of its synapses outside NP)
  neuron:<NP>   neurons with >=50% of their synapses in NP silenced (outputs zeroed);
                stimulated GRNs and readout neurons protected; NPs with >=20 neurons
  random:<n>:<seed>  n random neurons silenced (size control for neuron lesions)
Readouts: MN9 rate, SEZ_DA rates (9 neurons, E1b), PAM mean depolarisation (Vm).
One summary CSV per condition in results/e2/conds/ and full per-neuron rates
(Hz, all spiking neurons) in results/e2/rates/ (resumable; needs both).
"""
import argparse
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, rates, default_params, V783  # noqa: E402
from ffsim.readouts import load_annotations  # noqa: E402
from ffsim.lesions import neuropil_lesions, random_lesion  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'e2'
CONDS = OUT / 'conds'
RATES = OUT / 'rates'
RANDOM_SIZES = [50, 200, 1000, 5000, 28000]
VISUAL = ('ME_', 'LO_', 'LOP_', 'LA_', 'AME_', 'OCG')  # optic lobe / ocellar neuropils
MIN_NEURONS = 20


def readout_sets(ids):
    ann = load_annotations()
    pam = [f for f in ann[ann.cell_type.astype(str).str.startswith('PAM')].root_id if f in ids]
    sez = [f for f in pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id if f in ids]
    return pam, sez


def conditions(ids, protect, only=None, skip=(), order=('tissue', 'random', 'neuron'),
               random_sizes=RANDOM_SIZES):
    frac = pd.read_parquet(OUT / 'neuropil_fractions.parquet')
    nl = neuropil_lesions(frac, 0.5, protect)
    long = pd.read_parquet(OUT / 'conn_neuropil_frac.parquet')
    nps = sorted(set(long.neuropil.cat.categories) - {'UNASGD'})
    nps = [n for n in nps if not n.startswith(tuple(skip))]
    groups = {
        'tissue': [(f'tissue:{n}', {'tissue': n}) for n in nps],
        'random': [(f'random:{n}:{s}', {'slnc': random_lesion(ids, n, s, protect)})
                   for n in random_sizes for s in (0, 1)],
        'neuron': [(f'neuron:{n}', {'slnc': v}) for n, v in sorted(nl.items(), key=lambda kv: -len(kv[1]))
                   if len(v) >= MIN_NEURONS and not n.startswith(tuple(skip))],
    }
    c = [('intact', {})] + [x for g in order for x in groups[g]]
    if only:
        c = [x for x in c if any(x[0].startswith(o) for o in only)]
    return c, long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-run', type=int, default=15)
    ap.add_argument('--only', nargs='*', help='condition name prefixes')
    ap.add_argument('--skip-visual', action='store_true', help='drop optic lobe / ocellar neuropils')
    ap.add_argument('--order', nargs='+', default=['tissue', 'random', 'neuron'])
    ap.add_argument('--random-sizes', type=int, nargs='+', default=RANDOM_SIZES)
    ap.add_argument('--outdir', default='e2', help='results subdir (e.g. e2_confirm for reruns)')
    ap.add_argument('--n-intact', type=int, default=30)
    args = ap.parse_args()
    global OUT, CONDS, RATES
    OUT_RES = ROOT / 'results' / args.outdir
    CONDS, RATES = OUT_RES / 'conds', OUT_RES / 'rates'
    CONDS.mkdir(parents=True, exist_ok=True)
    RATES.mkdir(parents=True, exist_ok=True)

    ids_index = pd.read_csv(V783['path_comp'], index_col=0).index
    ids = set(ids_index)
    sugar = [f for f in NEU_SUGAR if f in ids]
    pam, sez = readout_sets(ids)
    protect = set(sugar) | {ID_MN9} | set(pam) | set(sez)
    conds, long = conditions(ids, protect, args.only, VISUAL if args.skip_visual else (),
                             args.order, args.random_sizes)
    n_rows = int(long.row.max()) + 1
    print(f'{len(conds)} conditions; {args.n_run} trials each', flush=True)

    params = dict(default_params)
    params['r_poi'] = 100 * Hz
    model = Model(sugar, params, data=V783, record_v=pam)
    for name, spec in conds:
        fname = f"{name.replace(':', '__')}.csv"
        path, rpath = CONDS / fname, RATES / fname
        if path.exists() and rpath.exists():
            continue
        t0 = time()
        n_run = args.n_intact if name == 'intact' else args.n_run
        params['n_run'] = n_run
        w_scale, slnc = None, spec.get('slnc', [])
        if 'tissue' in spec:
            sub = long[long.neuropil == spec['tissue']]
            w_scale = np.ones(n_rows, dtype=float)
            np.subtract.at(w_scale, sub.row.values, sub.frac.values.astype(float))
            w_scale = np.clip(w_scale, 0, 1)
        df_spk, df_v = model.run(w_scale=w_scale, slnc_ids=slnc, n_run=n_run)
        r = rates(df_spk, float(params['t_run'] / second), n_run)
        r.to_csv(rpath)
        dv = df_v.groupby('flywire_id').dv_mV.mean()
        mn9 = df_spk[df_spk.flywire_id == ID_MN9].groupby('trial').size().reindex(range(n_run), fill_value=0)
        sez_tr = df_spk[df_spk.flywire_id.isin(sez)].groupby('trial').size().reindex(range(n_run), fill_value=0) / len(sez)
        row = {
            'cond': name, 'type': name.split(':')[0], 'region': name.split(':')[1] if ':' in name else '',
            'n_silenced': len(slnc),
            'syn_removed': float(n_rows - w_scale.sum()) if w_scale is not None else 0.0,
            'n_run': n_run,
            'mn9': mn9.mean(), 'mn9_sd': mn9.std(ddof=0),
            'sez_da': sez_tr.mean(), 'sez_da_sd': sez_tr.std(ddof=0),
            'pam_dv': dv.mean(), 'pam_dv_max': dv.max(),
            'active_neurons': int((r.r > 0).sum()), 'spikes_per_trial': len(df_spk) / n_run,
        }
        row.update({f'sez_{f}': r.r.get(f, 0.0) for f in sez})
        pd.DataFrame([row]).to_csv(path, index=False)
        print(f"{name:22s} sil={len(slnc):6d} MN9={row['mn9']:5.1f} SEZ_DA={row['sez_da']:5.1f} "
              f"PAMdv={row['pam_dv']*1000:6.2f}uV act={row['active_neurons']:5d} {time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
