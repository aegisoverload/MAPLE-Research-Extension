"""Metadata diagnostics for MAPLE, not an automatic faithfulness classifier.

Run from the MAPLE repository root. Compares *requested* MAPLE metadata with
reference metadata, not attributes inferred from generated text.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "primary_research_area", "model_organism", "experimental_approach",
    "dominant_data_type", "research_focus_scale", "disease_mention",
    "sample_size", "research_goal",
]


def word_count(text):
    return len(str(text).split()) if pd.notna(text) else 0


def jsd(p, q):
    """Jensen-Shannon divergence in bits, for identically ordered categories."""
    p = np.asarray(p, dtype=np.float64).copy()
    q = np.asarray(q, dtype=np.float64).copy()
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distribution weights must be non-negative")
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl_div(a, b):
        positive = a > 0
        return np.sum(a[positive] * np.log2(a[positive] / b[positive]))

    return float(0.5 * kl_div(p, m) + 0.5 * kl_div(q, m))


def distribution(df, col):
    """Return value counts keyed by normalized metadata category."""
    values = df[col].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return values.value_counts()


def aligned_jsd(maple, reference, field):
    """Align category labels before comparing counts, including zero counts."""
    maple_counts = distribution(maple, field)
    reference_counts = distribution(reference, field)
    if maple_counts.empty or reference_counts.empty:
        return np.nan, len(maple_counts), len(reference_counts)
    categories = maple_counts.index.union(reference_counts.index)
    p = maple_counts.reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    q = reference_counts.reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    return jsd(p, q), len(maple_counts), len(reference_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maple", default="results/maple_30/PE_iter_1.csv")
    parser.add_argument("--augpe", default="results/augpe_30/PE_iter_1.csv")
    parser.add_argument("--reference", default="biorxiv/biorxiv_test_metadata.csv")
    parser.add_argument("--out-dir", default="results/evaluation/metadata_30")
    args = parser.parse_args()

    maple, augpe, ref = (pd.read_csv(p) for p in [args.maple, args.augpe, args.reference])
    for label, df in [("MAPLE", maple), ("AugPE", augpe), ("reference", ref)]:
        if "text" not in df.columns:
            raise ValueError(f"{label} missing text")
    if "word_count" not in maple.columns:
        raise ValueError("MAPLE output missing requested word_count")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for method, df in [("MAPLE", maple), ("AugPE", augpe)]:
        lengths = df["text"].map(word_count)
        record = {
            "method": method,
            "samples": len(df),
            "mean_words": float(lengths.mean()),
            "min_words": int(lengths.min()),
            "max_words": int(lengths.max()),
            "empty_texts": int((lengths == 0).sum()),
        }
        if method == "MAPLE":
            targets = pd.to_numeric(df["word_count"], errors="coerce")
            valid = targets.notna() & (targets > 0) & (lengths > 0)
            errors = (lengths[valid] - targets[valid]).abs()
            record.update({
                "n_valid_word_targets": int(valid.sum()),
                "mean_absolute_word_error": float(errors.mean()) if len(errors) else None,
                "mean_relative_word_error": float((errors / targets[valid]).mean()) if len(errors) else None,
                "fraction_within_20_percent": float(((errors / targets[valid]) <= 0.2).mean()) if len(errors) else None,
                "mean_signed_word_error": float((lengths[valid] - targets[valid]).mean()) if len(errors) else None,
            })
            sample_rows = pd.DataFrame({
                "target_words": targets,
                "actual_words": lengths,
                "absolute_error": (lengths - targets).abs(),
            })
            sample_rows.to_csv(out / "maple_word_count_rows.csv", index=False, encoding="utf-8-sig")
        records.append(record)
    pd.DataFrame(records).to_csv(out / "length_summary.csv", index=False, encoding="utf-8-sig")

    jsd_rows = []
    for field in FIELDS:
        if field not in maple.columns or field not in ref.columns:
            print(f"Skipping {field}: column missing from MAPLE or reference")
            continue
        score, n_categories_maple, n_categories_reference = aligned_jsd(maple, ref, field)
        if not np.isnan(score):
            jsd_rows.append({
                "metadata_field": field,
                "n_maple": int(maple[field].notna().sum()),
                "n_reference": int(ref[field].notna().sum()),
                "requested_metadata_jsd_bits": score,
                "n_categories_maple": n_categories_maple,
                "n_categories_reference": n_categories_reference,
            })
    pd.DataFrame(jsd_rows, columns=[
        "metadata_field", "n_maple", "n_reference", "requested_metadata_jsd_bits",
        "n_categories_maple", "n_categories_reference",
    ]).to_csv(out / "requested_metadata_jsd.csv", index=False, encoding="utf-8-sig")

    notes = {
        "warning": "Requested metadata JSD compares prompt-conditioning metadata with reference metadata; it does NOT measure generated-text metadata faithfulness or MAPLE-vs-AugPE quality.",
        "jsd": "JSD in bits on the UNION of observed non-null, non-empty category labels (zeros filled for missing categories); excludes missing values.",
        "word_count": "Whitespace splitting, consistent with generation metrics.",
        "small_sample_warning": "30 samples are for diagnostics only; interpret JSD with caution.",
    }
    (out / "README.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nLength summary:")
    print(pd.DataFrame(records).to_string(index=False))
    if jsd_rows:
        print("\nMAPLE requested metadata vs reference distribution (NOT text faithfulness):")
        print(pd.DataFrame(jsd_rows).to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
