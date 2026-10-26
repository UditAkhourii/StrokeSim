"""K4b: mirror test retry. K4-B (21 right GRNs) barely drove the circuit: right-side sugar GRNs
have about half the output synapses of the left set (median 169 vs 325; less complete
reconstruction). Here: ALL right-side GRNs of the same types (LB3b/c/d, LB4b), 100 Hz, 20 trials.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, default_params, V783  # noqa: E402
from experiments.k2_bilateral import CB0246_L, CB0246_R  # noqa: E402
from experiments.k4_robust_mirror import groups, run_set  # noqa: E402

OUT = ROOT / 'results' / 'k4' / 'mirror_all_right'


def main():
    ids = set(pd.read_csv(V783['path_comp'], index_col=0).index)
    a = pd.read_csv(ROOT / 'data' / 'annotations' / 'neuron_annotations.tsv', sep='\t',
                    usecols=['root_id', 'cell_type', 'side'], low_memory=False).drop_duplicates('root_id')
    right = sorted(a.root_id[(a.side == 'right') & a.cell_type.isin(['LB3b', 'LB3c', 'LB3d', 'LB4b']) & a.root_id.isin(ids)])
    print(f'{len(right)} right GRNs', flush=True)
    pump, pos = groups()
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    model = Model(right, p, data=V783)
    run_set(model, {'intact': ([], None), 'CB0246_L': ([CB0246_L], None), 'CB0246_R': ([CB0246_R], None)},
            20, OUT, float(p['t_run'] / second), pump, pos)


if __name__ == '__main__':
    main()
