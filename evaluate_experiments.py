"""Lightweight MAPLE vs AugPE prototype evaluator (not a DP audit or MAUVE).

Run at the MAPLE repository root:
  python evaluate_experiments.py --maple results/compare_maple --augpe results/compare_augpe
Optional embedding-based semantic diversity:
  python evaluate_experiments.py --maple results/compare_maple --augpe results/compare_augpe --semantic
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", str(text), flags=re.UNICODE))


def lexical_similarity(texts):
    if len(texts) < 2:
        return None
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words='english').fit_transform(texts)
    except ValueError:
        return None
    similarities = cosine_similarity(matrix)
    return float(similarities[np.triu_indices(len(texts), k=1)].mean())


def normalize(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


def quality_rows(method, iteration, frame, previous, semantic_model=None):
    texts = frame['text'].fillna('').astype(str).tolist()
    cleaned = [normalize(t) for t in texts]
    lengths = np.array([count_words(t) for t in texts], dtype=float)
    prior_set = set(previous['text'].fillna('').astype(str).map(normalize)) if previous is not None else set()
    retained = sum(t in prior_set for t in cleaned) if previous is not None else None
    heading_re = re.compile(r'(?im)^\s*(?:#{1,5}\s*)?(?:\*\*)?\s*(?:title|abstract|keywords?)\s*(?:\*\*)?\s*[:：]')
    heading_count = sum(bool(heading_re.search(t)) for t in texts)
    result = {
        'method': method,
        'iteration': iteration,
        'num_samples': len(texts),
        'empty_texts': sum(not t for t in cleaned),
        'exact_unique': len(set(cleaned)),
        'duplicate_count': len(cleaned) - len(set(cleaned)),
        'mean_words': round(float(lengths.mean()), 3),
        'min_words': int(lengths.min()),
        'max_words': int(lengths.max()),
        'format_marker_count': heading_count,
        'mean_pairwise_tfidf_cosine': lexical_similarity(texts),
        'retained_from_previous': retained,
        'new_since_previous': len(texts) - retained if retained is not None else None,
    }
    if 'word_count' in frame:
        targets = pd.to_numeric(frame['word_count'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(targets) & (targets > 0)
        result['target_words_mean'] = round(float(targets[valid].mean()), 3) if valid.any() else None
        result['mean_absolute_word_error'] = round(float(np.abs(lengths[valid] - targets[valid]).mean()), 3) if valid.any() else None
        result['mean_relative_word_error'] = round(float((np.abs(lengths[valid] - targets[valid]) / targets[valid]).mean()), 5) if valid.any() else None
    else:
        result['target_words_mean'] = None
        result['mean_absolute_word_error'] = None
        result['mean_relative_word_error'] = None
    if semantic_model is not None and len(texts) >= 2:
        embeddings = semantic_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sims = embeddings @ embeddings.T
        result['mean_pairwise_semantic_cosine'] = round(float(sims[np.triu_indices(len(texts), k=1)].mean()), 5)
    else:
        result['mean_pairwise_semantic_cosine'] = None
    return result


def evaluate_method(label, folder, semantic_model):
    frames = []
    rows = []
    previous = None
    for idx in (0, 1):
        path = folder / f'PE_iter_{idx}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing expected output: {path}')
        df = pd.read_csv(path)
        if 'text' not in df:
            raise ValueError(f'Missing text column: {path}')
        frames.append(df)
        rows.append(quality_rows(label, idx, df, previous, semantic_model))
        previous = df
    return rows


def main():
    parser = argparse.ArgumentParser(description='Prototype MAPLE vs AugPE output diagnostics')
    parser.add_argument('--maple', required=True, type=Path, help='MAPLE output directory')
    parser.add_argument('--augpe', required=True, type=Path, help='AugPE output directory')
    parser.add_argument('--out', default='results/evaluation', type=Path)
    parser.add_argument('--semantic', action='store_true', help='Compute optional sentence embedding diversity')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    sem = None
    if args.semantic:
        from sentence_transformers import SentenceTransformer
        sem = SentenceTransformer(args.embedding_model, device='cpu')
    rows = evaluate_method('MAPLE', args.maple, sem) + evaluate_method('AugPE', args.augpe, sem)
    report = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / 'quality_metrics.csv', index=False, encoding='utf-8-sig')
    (args.out / 'quality_metrics.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('\n=== Output diagnostics (not MAUVE, not a DP proof) ===')
    print(report.to_string(index=False))
    print(f'\nSaved: {args.out / "quality_metrics.csv"}')
    print('\nCAUTION: 4 samples per group. Treat all comparisons as diagnostics, not statistical evidence.')


if __name__ == '__main__':
    main()
