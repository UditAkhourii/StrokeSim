"""K6b: male CNS at FlyWire-like activity. K6 ran hot (about 18k active neurons and
serotonin cells near 68 Hz at scale 1.0, vs about 400 and 11 Hz in FlyWire). Lower the global
weight scale (0.3 to 0.6) to find the FlyWire-like regime; run the K6 lesion set there.
Sugar both sides, 10 trials.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brian2 import Hz, second  # noqa: E402
from ffsim.sim import Model, default_params  # noqa: E402
from experiments.k6_malecns import DATA, OUT, cells, lesions, run_set  # noqa: E402


def main():
    c = cells()
    p = dict(default_params)
    p.update(r_poi=100 * Hz)
    n_con = len(pd.read_parquet(DATA['path_con'], columns=['Presynaptic_Index']))
    model = Model(c['sugar_L'] + c['sugar_R'], p, data=DATA)
    conds = {}
    for s in [0.3, 0.4, 0.5, 0.6]:
        for k, les in lesions(c).items():
            conds[f'x{s}_{k}'] = (les, np.full(n_con, s))
    run_set(model, conds, 10, OUT / 'calibrated', float(p['t_run'] / second), c)


if __name__ == '__main__':
    main()
