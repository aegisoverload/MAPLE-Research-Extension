"""Auditable zCDP diagnostic for MAPLE nearest-neighbor Gaussian voting.

This is accounting under explicit assumptions, NOT a proof of end-to-end DP.
Run at the MAPLE repo root; requires Python standard library only.
"""
import argparse
import csv
import json
import math
from pathlib import Path

OFFICIAL_SIGMA = {1.0: 6.869194837509232, 2.0: 4.7482491352519505, 4.0: 2.5787580697142025}
OFFICIAL_ROUNDS = {1.0: 3, 2.0: 5, 4.0: 5}  # initialization + PE_T


def eps_from_rho(rho, delta):
    if rho == 0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def rho_from_eps(eps, delta):
    # Inverse of the standard conservative zCDP -> (eps, delta)-DP conversion.
    L = math.log(1.0 / delta)
    return (math.sqrt(L + eps) - math.sqrt(L)) ** 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default='results/evaluation/privacy_accounting')
    p.add_argument('--num-private', type=int, default=500)
    p.add_argument('--population-n', type=int, default=28846,
                   help='Population used to set delta; 500 is a sampled experiment, not the full population')
    p.add_argument('--delta', type=float, default=None)
    p.add_argument('--epsilon-target', type=float, default=4.0)
    p.add_argument('--PE_T', type=int, default=2)
    p.add_argument('--sigma', type=float, default=OFFICIAL_SIGMA[4.0])
    p.add_argument('--aim-rho', type=float, default=None,
                   help='CERTIFIED AIM zCDP cost, if verified from source/provenance')
    p.add_argument('--aim-budget-fraction', type=float, default=None,
                   help='Illustrative assumption only (e.g. 0.1); NOT evidence of AIM cost')
    p.add_argument('--adjacency', choices=['add_remove', 'replace_one'], default='add_remove')
    p.add_argument('--release-voting-scores', action='store_true',
                   help='Set if saved per-row voting numbers will be released, not merely used internally')
    args = p.parse_args()
    if args.num_private <= 0 or args.population_n <= 1 or args.PE_T < 0 or args.sigma <= 0:
        p.error('invalid sample counts, rounds, or sigma')
    delta = args.delta if args.delta is not None else 1.0 / args.population_n
    if not 0 < delta < 1 or args.epsilon_target <= 0:
        p.error('delta must be between zero and one, target epsilon positive')
    if args.aim_rho is not None and args.aim_budget_fraction is not None:
        p.error('specify only one of --aim-rho and --aim-budget-fraction')
    if args.aim_rho is not None and args.aim_rho < 0:
        p.error('--aim-rho cannot be negative')
    if args.aim_budget_fraction is not None and not 0 <= args.aim_budget_fraction <= 1:
        p.error('AIM budget fraction must be in [0,1]')

    # Fixed candidate set independent of private data, each private point casts exactly one vote:
    # add/remove-one: L2 histogram sensitivity <=1;
    # replace-one: one bin decreases, another increases, L2 sensitivity <= sqrt(2).
    sensitivity = 1.0 if args.adjacency == 'add_remove' else math.sqrt(2.0)
    rounds = args.PE_T + 1
    rho_vote_one = sensitivity ** 2 / (2.0 * args.sigma ** 2)
    rho_votes = rounds * rho_vote_one
    rho_budget = rho_from_eps(args.epsilon_target, delta)

    if args.aim_rho is not None:
        rho_aim, aim_status = args.aim_rho, 'supplied_not_verified_by_script'
    elif args.aim_budget_fraction is not None:
        rho_aim, aim_status = rho_budget * args.aim_budget_fraction, 'illustrative_fraction_not_verified'
    else:
        rho_aim, aim_status = None, 'unknown'

    rho_total = rho_votes + rho_aim if rho_aim is not None else None
    remaining = rho_budget - rho_aim if rho_aim is not None else None
    sigma_needed = (sensitivity * math.sqrt(rounds / (2.0 * remaining))) if remaining is not None and remaining > 0 else None

    report = dict(
        status='CONDITIONAL_ACCOUNTING_ONLY_NOT_CERTIFIED_DP',
        epsilon_target=args.epsilon_target, delta=delta,
        population_n=args.population_n, private_subset_n=args.num_private,
        adjacency=args.adjacency, histogram_l2_sensitivity=sensitivity,
        voting_rounds=rounds, noise_std=args.sigma,
        rho_vote_per_round=rho_vote_one, rho_votes_total=rho_votes,
        epsilon_votes_only_conservative=eps_from_rho(rho_votes, delta),
        rho_budget_conservative=rho_budget,
        aim_rho=rho_aim, aim_status=aim_status,
        rho_total_conditional=rho_total,
        epsilon_total_conditional=eps_from_rho(rho_total, delta) if rho_total is not None else None,
        within_target_conditional=(rho_total <= rho_budget) if rho_total is not None else None,
        minimum_sigma_conditional=sigma_needed,
        notes=[
          'epsilon is a conservative bound via eps = rho + 2 sqrt(rho log(1/delta)); official aim/accounting.py uses a tighter conversion.',
          'AIM cost must be established from the provenance of the supplied AIM metadata file; the name eps4 does not establish its rho.',
          'Subset-only privacy and full 28846-person privacy are distinct neighboring-dataset claims; choice of delta is a policy/analysis choice, not automatically 1/500.',
          'Fixed public/DP-derived candidates and one vote per person required for stated histogram sensitivity.',
          'Prior-round candidate adaptivity is compatible with sequential composition if each round is DP.',
          'Input private texts or embeddings must not be directly released; check all intermediate outputs and few-shot data provenance.',
          'NumPy pseudorandom Gaussian is not a cryptographically secure DP implementation.',
          'Potential reuse of training-derived AIM metadata and overlapping validation examples needs audited composition and provenance.',
          'Public release of per-row noisy voting scores needs the accounted mechanism and no auxiliary disclosure.',
        ],
        release_voting_scores=args.release_voting_scores,
        official_eps4_sigma=OFFICIAL_SIGMA[4.0],
        official_eps4_voting_rounds=OFFICIAL_ROUNDS[4.0],
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'privacy_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out/'privacy_report.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['metric','value'])
        for k,v in report.items():
            if k != 'notes': writer.writerow([k,v])
    for k,v in report.items():
        if k != 'notes': print(f'{k}: {v}')
    print('\nCAVEAT: conditional zCDP arithmetic only; NOT an end-to-end DP certification.')
    print(f'Saved: {out.resolve()}')

if __name__ == '__main__':
    main()
