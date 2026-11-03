"""Fast, conservative MAPLE DP implementation audit (NOT a DP certificate).
Run from repository root: python audit_maple_dp.py --source-dir biorxiv/MAPLE --run-dir results/maple_dp_secure_smoke
"""
import argparse, json, math, re
from pathlib import Path

S = 2.5787580697142025

def eps_bound(rho, delta):
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))

def inspect(src, run, epsilon, population):
    mini_path = src / 'mini_experiment.py'
    maple_path = src / 'MAPLE_windows.py'
    mini = mini_path.read_text(encoding='utf-8')
    maple = maple_path.read_text(encoding='utf-8')
    cfg_path = run / 'config.json'
    config = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.is_file() else {}
    t = int(config.get('PE_T', 2))
    sigma = float(config.get('noise_multiplier', S))
    rounds = t + 1
    delta = 1.0 / population
    rho1 = rounds / (2 * sigma**2) if sigma > 0 else float('inf')
    rho2 = 2 * rho1
    report = {
        'status': 'AUDIT_DIAGNOSTIC_NOT_DP_CERTIFICATION',
        'source_files': [str(mini_path), str(maple_path)],
        'run_config_found': bool(config),
        'run_config': config,
        'delta_assumed': delta,
        'vote_rounds': rounds,
        'gaussian_sigma': sigma,
        'add_remove_voting_only_rho': rho1,
        'replace_one_voting_only_rho': rho2,
        'add_remove_voting_only_epsilon_conservative': eps_bound(rho1, delta),
        'replace_one_voting_only_epsilon_conservative': eps_bound(rho2, delta),
        'aim_provenance_verified': False,
        'aim_rho': None,
        'seeded_private_subsampling': bool(re.search(r'priv_df\s*=\s*priv_df\.sample\(', mini)) and 'random_state=args.seed' in mini,
        'noise_uses_numpy_global_rng': bool(re.search(r'np\.random\.normal\s*\(', maple)),
        'noise_uses_secrets': 'secrets.SystemRandom' in maple,
        'exports_voting_column': "sub_cur_df['voting']" in maple and 'cur_df.to_csv(' in mini,
        'uses_dp_aim_for_initialization': 'aim_df.sample(' in mini,
        'uses_validation_in_context_examples': 'val_df.head(args.num_incontext)' in mini,
        'private_texts_written_to_disk_directly': 'UNKNOWN_STATIC_AUDIT',
        'fixed_size_sampling_requires_proof': True,
        'end_to_end_epsilon_proven': False,
        'blockers': [
            'Verify actual AIM rho and generation provenance',
            'Specify protected universe and fixed-size sampling / adjacency proof',
            'Verify secure DP noise implementation in THE EXACT code run',
            'Audit all files and outputs, including noisy voting statistics',
            'Check metadata annotations and in-context examples provenance'
        ]
    }
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir', default='biorxiv/MAPLE')
    p.add_argument('--run-dir', default='results/maple_dp_secure_smoke')
    p.add_argument('--epsilon', type=float, default=4)
    p.add_argument('--population', type=int, default=28846)
    p.add_argument('--out', default='results/evaluation/dp_audit.json')
    args=p.parse_args()
    result=inspect(Path(args.source_dir), Path(args.run_dir), args.epsilon, args.population)
    out=Path(args.out);out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for k in ['status','run_config_found','vote_rounds','gaussian_sigma','add_remove_voting_only_epsilon_conservative','replace_one_voting_only_epsilon_conservative','noise_uses_numpy_global_rng','noise_uses_secrets','seeded_private_subsampling','aim_provenance_verified','end_to_end_epsilon_proven']:
        print(f'{k}: {result[k]}')
    print('Saved:', out)
