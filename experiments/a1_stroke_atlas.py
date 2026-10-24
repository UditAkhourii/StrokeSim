"""A1: stroke analogue -- single-neuropil lesion atlas.

For each neuropil, silence every neuron with >=50% of its synapses there
(outgoing weights set to 0), stimulate sugar GRNs at 100 Hz (v783, default
W_syn) and measure:
  MI  = MN9 rate / intact            (feeding / motor index)
  DLI = SEZ_DA mean rate / intact    (reward stand-in, depression-like index)
  PAM_dv = mean PAM depolarisation   (secondary path-integrity metric)
  runaway = active neurons > 5000
Stimulated GRNs and readout neurons are spared from lesions.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from ffsim.readouts import load_annotations  # noqa: E402
from ffsim.lesion import load_neuropil, neuropil_lesion  # noqa: E402
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'a1'
MIN_NEURONS = 10
RUNAWAY_N = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-run', type=int, default=20)
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    (OUT / 'cond').mkdir(parents=True, exist_ok=True)
    (OUT / 'rates').mkdir(parents=True, exist_ok=True)

    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    a = load_annotations()
    sugar = [f for f in NEU_SUGAR if f in ids]
    sez_da = list(pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id)
    pam = [f for f in a[a.cell_type.astype(str).str.startswith('PAM')].root_id if f in ids]
    kc = set(a[a.cell_class == 'Kenyon_Cell'].root_id)
    spare = set(sugar) | set(sez_da) | {ID_MN9}

    npt = load_neuropil()
    sizes = npt[npt.frac >= 0.5].groupby('neuropil').size()
    neuropils = ['INTACT'] + sorted(sizes[sizes >= MIN_NEURONS].index.drop('None', errors='ignore'))

    params = dict(default_params)
    params.update(r_poi=100 * Hz, n_run=args.n_run)
    for npl in neuropils:
        if args.only and npl not in args.only:
            continue
        f_out = OUT / 'cond' / f'{npl}.csv'
        if f_out.exists():
            continue
        les = [] if npl == 'INTACT' else neuropil_lesion(npl, npt, spare=spare)
        print(f'>>> {npl}: lesion {len(les)} neurons', flush=True)
        df, dv = run(sugar, params, slnc_ids=les, data=V783, record_v=pam, verbose=False)
        rt = rates(df, float(params['t_run'] / second), args.n_run)
        rt.to_csv(OUT / 'rates' / f'{npl}.csv')  # all neurons, so readouts can be redefined later
        r = rt.r
        act = r[r > 1.0]
        pd.DataFrame([{
            'neuropil': npl, 'n_lesioned': len(les),
            'MN9_Hz': r.get(ID_MN9, 0.0),
            'SEZ_DA_Hz': r.reindex(sez_da).fillna(0).mean(),
            'PAM_dv_mV': dv.groupby('flywire_id').dv_mV.mean().mean(),
            'active_total': len(act),
            'KC_frac': len(set(act.index) & kc) / len(kc),
        }]).to_csv(f_out, index=False)

    res = pd.concat([pd.read_csv(p) for p in sorted((OUT / 'cond').glob('*.csv'))], ignore_index=True)
    base = res.set_index('neuropil').loc['INTACT']
    res['MI'] = res.MN9_Hz / base.MN9_Hz
    res['DLI'] = res.SEZ_DA_Hz / base.SEZ_DA_Hz
    res['PAM_dv_rel'] = res.PAM_dv_mV / base.PAM_dv_mV if base.PAM_dv_mV else float('nan')
    res['dissociation'] = res.MI - res.DLI   # >0: reward lost more than motor
    res['runaway'] = res.active_total > RUNAWAY_N
    res.sort_values('dissociation', ascending=False).to_csv(OUT / 'atlas.csv', index=False)
    print(res.sort_values('dissociation', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
