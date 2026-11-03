"""Conditional MAPLE Gaussian voting zCDP accountant and sigma calibrator.

Does not certify end-to-end DP. AIM cost, dataset adjacency, sampling,
Gaussian implementation and all releases require independent verification.
"""
import argparse
import json
import math
from pathlib import Path


def epsilon_from_rho(rho, delta):
    if rho < 0 or not 0 < delta < 1:
        raise ValueError('Invalid rho/delta')
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def rho_from_epsilon(epsilon, delta):
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError('Invalid epsilon/delta')
    L = math.log(1 / delta)
    return (math.sqrt(L + epsilon) - math.sqrt(L)) ** 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epsilon-target', type=float, default=4.0)
    p.add_argument('--population-n', type=int, default=28846)
    p.add_argument('--num-private', type=int, default=500)
    p.add_argument('--delta', type=float)
    p.add_argument('--PE_T', type=int, default=2)
    p.add_argument('--adjacency', choices=['replace_one', 'add_remove'], default='replace_one')
    p.add_argument('--aim-rho', type=float, help='Verified AIM rho only; supply with provenance')
    p.add_argument('--aim-budget-fraction', type=float,
                   help='Hypothetical AIM share of conservative rho budget, NOT proof of provenance')
    p.add_argument('--sigma', type=float, help='Optional existing sigma to check')
    p.add_argument('--sigma-margin', type=float, default=1.05,
                   help='Factor above calibrated minimum, default 1.05')
    p.add_argument('--release-voting-scores', action='store_true')
    p.add_argument('--output-dir', default='results/evaluation/privacy_calibrated')
    args = p.parse_args()

    if args.population_n < 2 or args.num_private < 1 or args.PE_T < 0:
        p.error('Invalid data sizes or PE_T')
    if args.sigma_margin < 1 or (args.sigma is not None and args.sigma <= 0):
        p.error('sigma must be positive and margin must be >= 1')
    if args.aim_rho is not None and args.aim_budget_fraction is not None:
        p.error('Supply only one of --aim-rho or --aim-budget-fraction')
    if args.aim_rho is not None and args.aim_rho < 0:
        p.error('--aim-rho must be nonnegative')
    if args.aim_budget_fraction is not None and not 0 <= args.aim_budget_fraction <= 1:
        p.error('AIM fraction must lie between 0 and 1')

    delta = args.delta if args.delta is not None else 1 / args.population_n
    if not 0 < delta < 1:
        p.error('delta must be in (0,1)')
    rho_budget = rho_from_epsilon(args.epsilon_target, delta)
    if args.aim_rho is not None:
        rho_aim = args.aim_rho
        aim_status = 'user_supplied_cost_not_verified_by_program'
    elif args.aim_budget_fraction is not None:
        rho_aim = rho_budget * args.aim_budget_fraction
        aim_status = 'hypothetical_budget_fraction_NOT_VERIFIED'
    else:
        rho_aim = None
        aim_status = 'unknown_no_calibration_possible'

    K = args.PE_T + 1
    delta_l2 = math.sqrt(2) if args.adjacency == 'replace_one' else 1.0
    rho_available = rho_budget - rho_aim if rho_aim is not None else None
    if rho_available is not None and rho_available <= 0:
        p.error('No positive voting budget remains after AIM allocation')

    min_sigma = delta_l2 * math.sqrt(K / (2 * rho_available)) if rho_available is not None else None
    recommended_sigma = min_sigma * args.sigma_margin if min_sigma is not None else None

    def costs(sigma):
        if sigma is None:
            return None
        rho_votes = K * delta_l2 ** 2 / (2 * sigma ** 2)
        total = rho_votes + rho_aim if rho_aim is not None else None
        return {
            'sigma': sigma,
            'rho_votes': rho_votes,
            'epsilon_votes_only_conservative': epsilon_from_rho(rho_votes, delta),
            'rho_total_conditional': total,
            'epsilon_total_conditional': epsilon_from_rho(total, delta) if total is not None else None,
            'within_target_conditional': total <= rho_budget if total is not None else None,
        }

    report = {
        'status': 'CONDITIONAL_ONLY_NOT_CERTIFIED_END_TO_END_DP',
        'epsilon_target': args.epsilon_target,
        'delta': delta,
        'population_n_for_delta': args.population_n,
        'private_subset_n': args.num_private,
        'adjacency': args.adjacency,
        'histogram_l2_sensitivity_assumed': delta_l2,
        'PE_T': args.PE_T,
        'voting_rounds': K,
        'rho_budget_conservative': rho_budget,
        'aim_rho_assumed': rho_aim,
        'aim_status': aim_status,
        'rho_available_for_votes': rho_available,
        'minimum_sigma_conditional': min_sigma,
        'sigma_margin': args.sigma_margin,
        'recommended_sigma_conditional': recommended_sigma,
        'existing_sigma_check': costs(args.sigma),
        'recommended_sigma_check': costs(recommended_sigma),
        'release_voting_scores': args.release_voting_scores,
        'caveats': [
            'Fixed private subset sampling is not accounted for; replace-one analysis is conditional on a fixed private dataset.',
            'AIM metadata provenance and rho must be validated; a fraction is an illustrative assumption.',
            'The histogram sensitivity presumes each record contributes one nearest-neighbor vote to private-independent or DP-derived candidates.',
            'Any unprotected private-dependent outputs invalidate this composition argument.',
            'Secrets-based Gaussian sampling is not, by itself, a formally audited floating-point DP mechanism.',
            'Previously generated experiments are not retroactively made private by this calibration.',
        ],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'privacy_calibration.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for key, value in report.items():
        if key != 'caveats':
            print(f'{key}: {value}')
    print('CAVEAT: conditional math only; NOT an end-to-end DP certification.')
    print(f'Saved: {out / "privacy_calibration.json"}')


if __name__ == '__main__':
    main()
