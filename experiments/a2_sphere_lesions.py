"""A2: focal stroke -- spherical lesions centred on neuropil centroids.

Core: neurons whose annotation position point (4x4x40 nm voxels -> um) lies within
radius r of the centre are silenced (outgoing weights 0). Penumbra (optional): the
shell r < d <= PEN_K * r has outgoing weights scaled by PEN_FACTOR. Controls: for each
sphere condition (penumbra off), the same number of neurons drawn at random from the
whole brain (spatially incoherent). Sugar GRNs at 100 Hz, v783, same readouts as A1.
Stimulated GRNs and readout neurons are spared.
"""
import argparse
import sys
import zlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import run, rates, default_params, V783  # noqa: E402
from ffsim.readouts import load_annotations  # noqa: E402
from ffsim.lesion import (load_neuropil, load_positions, neuropil_centroids,  # noqa: E402
                          sphere_lesion, random_size_matched)
from experiments.e0_baseline import NEU_SUGAR, ID_MN9  # noqa: E402

OUT = ROOT / 'results' / 'a2'
RADII = [20, 30, 45, 60]
PEN_K = 1.5               # penumbra outer radius / core radius
PEN_FACTOR = 0.5          # penumbra outgoing-weight scale
N_RANDOM_SEEDS = 1
SEC_PER_TRIAL = 110 / 20   # s per trial (~110 s per condition at 20 trials)
RUNAWAY_N = 5000


def conditions(centers, pos, spare, n_seeds):
    """List of dicts: cond, center, radius, penumbra, core, pen."""
    out = [dict(cond='INTACT', center='', radius=0, penumbra=False, core=[], pen=[])]
    for npl, c in centers.iterrows():
        xyz = c[['x', 'y', 'z']].to_numpy(float)
        for r in RADII:
            core, pen = sphere_lesion(xyz, r, pos, penumbra_k=PEN_K, spare=spare)
            if not core:
                continue
            out.append(dict(cond=f'{npl}_r{r}', center=npl, radius=r, penumbra=False, core=core, pen=[]))
            out.append(dict(cond=f'{npl}_r{r}_pen', center=npl, radius=r, penumbra=True, core=core, pen=pen))
            for s in range(n_seeds):
                rnd = random_size_matched(len(core), exclude=spare, seed=zlib.crc32(f"{npl}_{r}_{s}".encode()), pool=pos.index)
                out.append(dict(cond=f'RAND_{npl}_r{r}_s{s}', center=f'RAND_{npl}', radius=r,
                                penumbra=False, core=rnd, pen=[]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-run', type=int, default=20)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--seeds', type=int, default=N_RANDOM_SEEDS)
    ap.add_argument('--dry-run', action='store_true', help='print plan and lesion sizes only')
    args = ap.parse_args()
    (OUT / 'cond').mkdir(parents=True, exist_ok=True)

    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    a = load_annotations()
    sugar = [f for f in NEU_SUGAR if f in ids]
    sez_da = list(pd.read_csv(ROOT / 'results' / 'e1b' / 'sez_da_ids.csv').flywire_id)
    pam = [f for f in a[a.cell_type.astype(str).str.startswith('PAM')].root_id if f in ids]
    kc = set(a[a.cell_class == 'Kenyon_Cell'].root_id)
    spare = set(sugar) | set(sez_da) | {ID_MN9}

    pos = load_positions()
    centers = neuropil_centroids(load_neuropil(), pos)
    centers.round(1).to_csv(OUT / 'centers.csv')
    conds = conditions(centers, pos, spare, args.seeds)
    if args.only:
        conds = [c for c in conds if c['cond'] in args.only]
    todo = [c for c in conds if not (OUT / 'cond' / f"{c['cond']}.csv").exists()]
    est = len(todo) * args.n_run * SEC_PER_TRIAL
    print(f'{len(conds)} conditions ({len(centers)} centres x {len(RADII)} radii x '
          f'[core, core+penumbra, {args.seeds} random] + INTACT); {len(todo)} to run; '
          f'est. {est / 3600:.1f} h at {args.n_run} trials', flush=True)
    if args.dry_run:
        for c in conds:
            print(f"  {c['cond']:<24} core {len(c['core']):>6}  pen {len(c['pen']):>6}")
        return

    params = dict(default_params)
    params.update(r_poi=100 * Hz, n_run=args.n_run)
    for c in todo:
        f_out = OUT / 'cond' / f"{c['cond']}.csv"
        print(f">>> {c['cond']}: core {len(c['core'])}, penumbra {len(c['pen'])}", flush=True)
        df, dv = run(sugar, params, slnc_ids=c['core'], data=V783, record_v=pam, verbose=False,
                     scale_ids=c['pen'], scale_factor=PEN_FACTOR)
        rt = rates(df, float(params['t_run'] / second), args.n_run)
        (OUT / 'rates').mkdir(parents=True, exist_ok=True)
        rt.to_csv(OUT / 'rates' / f"{c['cond']}.csv")
        r = rt.r
        act = r[r > 1.0]
        pd.DataFrame([{
            'cond': c['cond'], 'center': c['center'], 'radius_um': c['radius'],
            'penumbra': c['penumbra'], 'n_lesioned': len(c['core']), 'n_penumbra': len(c['pen']),
            'MN9_Hz': r.get(ID_MN9, 0.0),
            'SEZ_DA_Hz': r.reindex(sez_da).fillna(0).mean(),
            'PAM_dv_mV': dv.groupby('flywire_id').dv_mV.mean().mean(),
            'active_total': len(act),
            'KC_frac': len(set(act.index) & kc) / len(kc),
        }]).to_csv(f_out, index=False)

    res = pd.concat([pd.read_csv(p) for p in sorted((OUT / 'cond').glob('*.csv'))], ignore_index=True)
    base = res.set_index('cond').loc['INTACT']
    res['MI'] = res.MN9_Hz / base.MN9_Hz
    res['DLI'] = res.SEZ_DA_Hz / base.SEZ_DA_Hz
    res['PAM_dv_rel'] = res.PAM_dv_mV / base.PAM_dv_mV if base.PAM_dv_mV else float('nan')
    res['dissociation'] = res.MI - res.DLI   # >0: reward lost more than motor
    res['runaway'] = res.active_total > RUNAWAY_N
    res.sort_values('dissociation', ascending=False).to_csv(OUT / 'sphere_atlas.csv', index=False)
    print(res.sort_values('dissociation', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
