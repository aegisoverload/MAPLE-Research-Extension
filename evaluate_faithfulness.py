"""Experimental metadata faithfulness diagnostic for MAPLE and AugPE.

Train text->metadata classifiers on PUBLIC bioRxiv validation labels only.
Evaluate held-out validation performance, and flag unreliable fields.
Do not interpret inferred labels as ground truth or as a DP guarantee.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

FIELDS = [
    'primary_research_area', 'model_organism', 'experimental_approach',
    'dominant_data_type', 'research_focus_scale', 'disease_mention',
    'sample_size', 'research_goal',
]


def normalized(s):
    return s.astype('string').fillna('').str.strip().str.casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maple', default='results/maple_30/PE_iter_1.csv')
    ap.add_argument('--augpe', default='results/augpe_30/PE_iter_1.csv')
    ap.add_argument('--validation', default='biorxiv/biorxiv_valid_metadata.csv')
    ap.add_argument('--out-dir', default='results/evaluation/faithfulness_30')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-class-samples', type=int, default=5)
    ap.add_argument('--min-holdout-macro-f1', type=float, default=0.40)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    maple = pd.read_csv(args.maple)
    augpe = pd.read_csv(args.augpe)
    val = pd.read_csv(args.validation)
    for name, frame in [('MAPLE', maple), ('AugPE', augpe), ('validation', val)]:
        if 'text' not in frame:
            raise ValueError(f'{name} CSV missing text column')
    maple_text = maple['text'].fillna('').astype(str)
    augpe_text = augpe['text'].fillna('').astype(str)
    diagnostics, predictions, coverage = [], [], []

    for field in FIELDS:
        if field not in val or field not in maple:
            diagnostics.append({'field': field, 'status': 'missing label column'})
            continue
        train = val[['text', field]].dropna().copy()
        train['text'] = train['text'].astype(str)
        train['label'] = normalized(train[field])
        train = train[(train['text'].str.strip() != '') & (train['label'] != '')]
        counts = train['label'].value_counts()
        eligible = counts[counts >= args.min_class_samples].index
        train = train[train['label'].isin(eligible)]
        if train['label'].nunique() < 2 or len(train) < 40:
            diagnostics.append({'field': field, 'status': 'insufficient labeled validation data',
                                'n_train_candidates': len(train)})
            continue
        stratify = train['label'] if train['label'].value_counts().min() >= 2 else None
        xtrain, xtest, ytrain, ytest = train_test_split(
            train['text'], train['label'], test_size=0.25,
            random_state=args.seed, stratify=stratify)
        model = make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                            min_df=2, max_features=30000, sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight='balanced'))
        model.fit(xtrain, ytrain)
        predicted = model.predict(xtest)
        macro_f1 = float(f1_score(ytest, predicted, average='macro', zero_division=0))
        accepted = macro_f1 >= args.min_holdout_macro_f1
        diagnostics.append({
            'field': field, 'status': 'diagnostic_only' if accepted else 'unreliable_classifier',
            'validation_samples': len(train), 'classes': int(train['label'].nunique()),
            'heldout_accuracy': float(accuracy_score(ytest, predicted)),
            'heldout_balanced_accuracy': float(balanced_accuracy_score(ytest, predicted)),
            'heldout_macro_f1': macro_f1,
        })
        # Predictions are exploratory; flag weak fields rather than reporting
        # a definitive faithfulness score.
        for method, frame, text in [('MAPLE', maple, maple_text), ('AugPE', augpe, augpe_text)]:
            pred = model.predict(text)
            requested = normalized(frame[field]) if field in frame else pd.Series([''] * len(frame))
            supported = set(model.classes_)
            for i, (p, req) in enumerate(zip(pred, requested)):
                predictions.append({
                    'method': method, 'row': i, 'field': field,
                    'predicted_label': p,
                    'requested_label': req if method == 'MAPLE' else '',
                    'requested_label_in_classifier': bool(req in supported) if method == 'MAPLE' else None,
                    'predicted_matches_requested': bool(p == req) if method == 'MAPLE' and req in supported else None,
                    'classifier_reliable_enough_for_diagnostic': accepted,
                })
            if method == 'MAPLE':
                in_scope = requested.isin(supported)
                match = (pd.Series(pred) == requested.reset_index(drop=True))
                coverage.append({
                    'field': field,
                    'classifier_accepted': accepted,
                    'requested_label_coverage': float(in_scope.mean()),
                    'n_scored': int(in_scope.sum()),
                    'match_rate_on_supported_labels': float(match[in_scope.to_numpy()].mean()) if in_scope.any() else None,
                    'caution': 'Classifier predictions, not verified ground-truth faithfulness',
                })
    pd.DataFrame(diagnostics).to_csv(out / 'classifier_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(out / 'metadata_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(out / 'maple_match_diagnostic.csv', index=False, encoding='utf-8-sig')
    (out / 'README.json').write_text(json.dumps({
        'purpose': 'Exploratory text-to-metadata classification diagnostics only',
        'training_source': args.validation,
        'no_private_training_data_used': True,
        'important_limitations': [
            'Predictions are not verified annotations or proof of metadata faithfulness.',
            'Low macro-F1 fields must not be interpreted as reliable.',
            'MAPLE match rates are conditioned on labels represented in the classifier.',
            'AugPE has no requested metadata; its output is predictions only.',
            'Do not interpret these outputs as DP guarantees or publish private outputs.'
        ]}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nClassifier held-out diagnostics:')
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print('\nMAPLE tentative metadata match diagnostics:')
    print(pd.DataFrame(coverage).to_string(index=False))
    print(f'\nSaved to: {out}')


if __name__ == '__main__':
    main()
