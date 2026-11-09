"""DP metadata via the repo's AIM implementation (aim/generate.py), rho_AIM = 0.1 * rho_total."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_common import SCHEMA, WORK

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'aim'))
import accounting  # noqa: E402
import domain  # noqa: E402
import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epsilon', type=float, default=1.0)
    ap.add_argument('--aim-fraction', type=float, default=0.1)
    ap.add_argument('--num-gen', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(WORK / 'private.csv')[list(SCHEMA)]
    n = len(df)
    delta = 1.0 / n
    rho_total = accounting.zcdp_rho(args.epsilon, delta)
    rho_aim = args.aim_fraction * rho_total
    domains = {k: domain.CategoricalAttribute(v) for k, v in SCHEMA.items()}
    synth = generate.run(domains, df, epsilon=0.0, delta=0.0, rho=rho_aim, seed=args.seed,
                         num_records=args.num_gen, pgm_iters=1000)
    synth = synth[list(SCHEMA)]
    out = WORK / f'dp_metadata_eps{args.epsilon:g}.csv'
    synth.to_csv(out, index=False)

    def tvd(c):
        p = df[c].value_counts(normalize=True)
        q = synth[c].value_counts(normalize=True)
        return float(0.5 * (p.subtract(q, fill_value=0).abs().sum()))

    report = {'epsilon_total': args.epsilon, 'delta': delta, 'rho_total': rho_total,
              'rho_aim': rho_aim, 'aim_fraction': args.aim_fraction, 'n_private': n,
              'num_gen': args.num_gen, 'seed': args.seed,
              'marginal_tvd': {c: round(tvd(c), 4) for c in SCHEMA}}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
