
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from MAPLE_windows import (
    API,
    SamplingParams,
    Voting_histogram,
    get_next_iteration,
)

random.seed(42)
np.random.seed(42)

# Metadata matching the bioRxiv schema
metadata = {
    "primary_research_area": "Cancer Biology",
    "model_organism": "Human",
    "experimental_approach": "Computational / In Silico Analysis",
    "dominant_data_type": "Genomic",
    "research_focus_scale": "Molecular",
    "disease_mention": "Cancer",
    "sample_size": "Not Specified / Not Applicable",
    "research_goal": "Developing a method/tool",
    "word_count": 150,
}

# Toy private dataset (no real private information)
priv_df = pd.DataFrame({
    "text": [
        "This study investigates genetic mechanisms underlying tumor progression.",
        "We analyze genomic biomarkers associated with cancer development.",
        "Computational methods reveal molecular patterns in human tumors.",
        "Genomic analysis identifies mutations associated with cancer progression.",
    ],
    "PE.LABEL_ID": [0, 0, 0, 0],
})

# Non-private in-context examples
examples = [
    (
        metadata,
        "We investigated genomic patterns in human cancer samples "
        "using computational analysis. Our results identified "
        "molecular signatures associated with tumor progression."
    )
]

# Generate 4 initial candidates
initial_df = pd.DataFrame([metadata.copy() for _ in range(4)])

print("Loading Qwen...")
api = API(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=180,
    ),
    examples=examples,
    n_sample=1,
    METADATA=list(metadata.keys()),
)

print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

class EmbeddingAdapter:
    def __init__(self, model):
        self._model = model

vh = Voting_histogram(
    priv_df=priv_df,
    embedding=EmbeddingAdapter(embedding_model),
    batch_size=4,
    noise_multiplier=0,
)

# Step 1: RANDOM
print("\n=== RANDOM API ===")
cur_df = api.RANDOM_API(initial_df.copy())

# Step 2: Voting and Selection
print("\n=== Initial Voting ===")
cur_df = vh.get_voting_histogram(cur_df)
cur_df = get_next_iteration(
    cur_df,
    num_samples=2,
    selection_mode="rank",
)

print(cur_df[["voting", "text"]])

# Step 3: Variation
print("\n=== VARIATION API ===")
var_df = api.VARIATION_API(cur_df.copy())

# Step 4: Combine original and varied candidates
combined_df = pd.concat(
    [cur_df, var_df],
    ignore_index=True,
)

# Step 5: Voting and Selection again
print("\n=== Final Voting ===")
combined_df = vh.get_voting_histogram(combined_df)

final_df = get_next_iteration(
    combined_df,
    num_samples=2,
    selection_mode="rank",
)

print(final_df[["voting", "text"]])

final_df.to_csv(
    "maple_iteration_test.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nMAPLE iteration smoke test completed!")
