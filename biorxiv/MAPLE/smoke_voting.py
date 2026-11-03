
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from MAPLE_windows import Voting_histogram, get_next_iteration

print("Loading embedding model...")

# Use a smaller model for RTX 3070 / CPU testing.
embedding = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# MAPLE accesses embedding._model.encode()
class EmbeddingAdapter:
    def __init__(self, model):
        self._model = model

# Simulated private samples (not sensitive data)
priv_df = pd.DataFrame({
    "text": [
        "Cancer cells proliferate through genetic mutations.",
        "Tumor growth is influenced by immune responses.",
        "Neurons communicate through synaptic connections.",
        "Brain activity changes during memory formation."
    ],
    "PE.LABEL_ID": [0, 0, 0, 0]
})

# Synthetic candidates
syn_df = pd.DataFrame({
    "text": [
        "Genetic alterations contribute to cancer development.",
        "Neural circuits support memory and cognition.",
        "Agricultural productivity depends on soil quality."
    ]
})

vh = Voting_histogram(
    priv_df=priv_df,
    embedding=EmbeddingAdapter(embedding),
    batch_size=4,
    noise_multiplier=0
)

print("\nComputing voting histogram...")
voted_df = vh.get_voting_histogram(syn_df.copy())

print(voted_df[["text", "voting"]])

print("\nSelecting next generation...")
selected_df = get_next_iteration(
    voted_df,
    num_samples=2,
    selection_mode="rank"
)

print(selected_df[["text", "voting"]])

print("\nVoting smoke test completed.")
