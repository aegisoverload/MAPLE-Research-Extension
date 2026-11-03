
import pandas as pd

from MAPLE_windows import API, SamplingParams

# Simulated metadata
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

# A public example for in-context learning
examples = [
    (
        metadata,
        "This study investigates computational methods "
        "for identifying cancer-associated genetic "
        "patterns using genomic datasets."
    )
]

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

api = API(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    sampling_params=params,
    examples=examples,
    n_sample=1,
    METADATA=list(metadata.keys())
)

# Test RANDOM API
df = pd.DataFrame([metadata])

print("\n=== RANDOM API ===")
random_result = api.RANDOM_API(df.copy())
print(random_result["text"].iloc[0])

# Test VARIATION API
print("\n=== VARIATION API ===")
variation_result = api.VARIATION_API(
    random_result.copy()
)
print(variation_result["text"].iloc[0])

print("\nMAPLE API smoke test completed.")
