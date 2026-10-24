"""Build data/neuropil/neuron_neuropil_783.parquet from the FlyWire v783 Zenodo tables.

Input (Zenodo record 10676866, see scripts/download_data.sh):
  data/neuropil/per_neuron_neuropil_count_{pre,post}_783.feather
  external/Drosophila_brain_model/Completeness_783.csv  (defines the simulated v783 neurons)

Output: long table flywire_id, neuropil, count, frac where count = pre + post synapses of
the neuron in that neuropil and frac = count / total synapses of the neuron. Restricted to
neurons in Completeness_783. Used by ffsim.lesion for neuropil (stroke) lesion sets.

usage (from repo root): python scripts/build_neuropil_table.py [--out PATH] [--force]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'neuropil'
COMP = ROOT / 'external' / 'Drosophila_brain_model' / 'Completeness_783.csv'


def build():
    ids = set(pd.read_csv(COMP, index_col=0).index)
    parts = []
    for k in ['pre', 'post']:
        d = pd.read_feather(SRC / f'per_neuron_neuropil_count_{k}_783.feather')
        d = d.rename(columns={f'{k}_pt_root_id': 'flywire_id'})
        parts.append(d[d.flywire_id.isin(ids)])
    d = pd.concat(parts).groupby(['flywire_id', 'neuropil'], as_index=False)['count'].sum()
    d['frac'] = d['count'] / d.groupby('flywire_id')['count'].transform('sum')
    return d.sort_values(['flywire_id', 'neuropil']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(SRC / 'neuron_neuropil_783.parquet'))
    ap.add_argument('--force', action='store_true', help='overwrite an existing output file')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f'{out} exists; use --force to rebuild')
        return
    d = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)
    print(f'wrote {out}: {len(d)} rows, {d.flywire_id.nunique()} neurons')


if __name__ == '__main__':
    main()
