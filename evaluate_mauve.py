"""Compare MAPLE and AugPE with MAUVE using Sentence-T5 features.

Use >= 50 samples per generator for a preliminary study; hundreds/thousands
for interpretable research results. This uses the *same* held-out reference
texts and one common embedding model for both methods.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def texts(path):
    frame = pd.read_csv(path)
    if 'text' not in frame:
        raise ValueError(f'Missing text column: {path}')
    values = frame['text'].dropna().astype(str).str.strip()
    return [x for x in values if x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maple', default='results/compare_maple/PE_iter_1.csv')
    p.add_argument('--augpe', default='results/compare_augpe/PE_iter_1.csv')
    p.add_argument('--reference', default='biorxiv/biorxiv_test_metadata.csv')
    p.add_argument('--out', default='results/evaluation/mauve_metrics.csv')
    p.add_argument('--min-samples', type=int, default=30)
    p.add_argument('--model', default='sentence-transformers/sentence-t5-base')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.min_samples < 10:
        p.error('--min-samples must be at least 10')
    sets = {'MAPLE': texts(args.maple), 'AugPE': texts(args.augpe)}
    reference = texts(args.reference)
    for name, data in sets.items():
        print(f'{name}: {len(data)} generated texts')
    print(f'Reference: {len(reference)} texts')
    if min(len(sets['MAPLE']), len(sets['AugPE'])) < args.min_samples:
        print(f'Not computing MAUVE: at least {args.min_samples} generated samples per method required for this diagnostic run.')
        print('Your current 4-sample outputs are valid smoke tests, but a numerical MAUVE score would be misleading.')
        return

    import mauve
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device='cpu')
    n = min(len(sets['MAPLE']), len(sets['AugPE']), len(reference))
    rng = np.random.default_rng(args.seed)
    ref_samples = [reference[i] for i in rng.choice(len(reference), size=n, replace=False)]
    print(f'Embedding equal-size samples: n={n}')
    ref_features = model.encode(ref_samples, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
    records = []
    for name, data in sets.items():
        generated = [data[i] for i in np.random.default_rng(args.seed).choice(len(data), size=n, replace=False)]
        features = model.encode(generated, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
        score = mauve.compute_mauve(
            p_features=ref_features, q_features=features,
            num_buckets=max(2, n // 10), seed=args.seed,
            kmeans_num_redo=3, verbose=False)
        records.append({'method': name, 'samples': n, 'mauve': float(score.mauve), 'model': args.model})
        print(f'{name}: MAUVE={score.mauve:.4f}')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
