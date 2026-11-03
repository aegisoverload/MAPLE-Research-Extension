
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

from MAPLE_windows import (
    API, SamplingParams,
    Voting_histogram, get_next_iteration
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "biorxiv"

METADATA = [
    "primary_research_area",
    "model_organism",
    "experimental_approach",
    "dominant_data_type",
    "research_focus_scale",
    "disease_mention",
    "sample_size",
    "research_goal",
    "word_count",
]

# 1. Load official data
aim_files = list(DATA.glob("*aim*eps4.0*.csv"))
if len(aim_files) != 1:
    raise RuntimeError(f"Expected one AIM file: {aim_files}")

aim_df = pd.read_csv(aim_files[0])
priv_df = pd.read_csv(DATA / "biorxiv_train_metadata.csv")
val_df = pd.read_csv(DATA / "biorxiv_valid_metadata.csv")

# Check required columns
for name, df, required in [
    ("AIM", aim_df, METADATA),
    ("Private", priv_df, ["text"]),
    ("Validation", val_df, METADATA + ["text"]),
]:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

print("AIM:", aim_df.shape)
print("Private:", priv_df.shape)
print("Validation:", val_df.shape)

# 2. Small test subset
aim_df = aim_df.dropna(subset=METADATA).head(4).copy()
priv_df = priv_df.dropna(subset=["text"]).head(20).copy()
val_df = val_df.dropna(subset=METADATA + ["text"]).head(2).copy()

priv_df["PE.LABEL_ID"] = 0

examples = [
    (row[METADATA].to_dict(), str(row["text"]))
    for _, row in val_df.iterrows()
]

# 3. Initialize MAPLE API
api = API(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    ),
    examples=examples,
    n_sample=2,
    METADATA=METADATA,
)

embedding = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

class EmbeddingAdapter:
    def __init__(self, model):
        self._model = model

vh = Voting_histogram(
    priv_df=priv_df,
    embedding=EmbeddingAdapter(embedding),
    batch_size=8,
    noise_multiplier=2.5787580697142025
)

# 4. MAPLE Initialization
print("\n=== RANDOM API ===")
cur_df = api.RANDOM_API(aim_df.copy())

print("\n=== INITIAL VOTING ===")
cur_df = vh.get_voting_histogram(cur_df)
cur_df = get_next_iteration(
    cur_df, num_samples=2, selection_mode="rank"
)

# 5. One evolution round
print("\n=== VARIATION API ===")
var_df = api.VARIATION_API(cur_df.copy())

combined = pd.concat(
    [cur_df, var_df], ignore_index=True
)

print("\n=== FINAL VOTING ===")
combined = vh.get_voting_histogram(combined)

final_df = get_next_iteration(
    combined, num_samples=2, selection_mode="rank"
)

# 6. Save
out_path = ROOT / "maple_real_data_test.csv"
final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
for _, row in final_df.iterrows():
    target = int(row["word_count"])
    actual = len(row["text"].split())

    print(
        f"Target words: {target}, "
        f"Actual words: {actual}, "
        f"Difference: {actual - target}"
    )

print(final_df[["voting", "text"]].to_string(index=False))
print(f"\nSaved to: {out_path}")
print("Real-data MAPLE smoke test completed!")
