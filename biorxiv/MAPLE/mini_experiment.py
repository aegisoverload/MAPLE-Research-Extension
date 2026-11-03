
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from MAPLE_windows import (
    API,
    SamplingParams,
    Voting_histogram,
    get_next_iteration,
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


class EmbeddingAdapter:
    def __init__(self, model):
        self._model = model

class AugPEAPI(API):
    """AugPE prompts using the existing MAPLE generation backend."""

    def _get_api_prompts(self, metadata_df, api_type="random"):
        assert api_type in ["random", "variation"]

        prompts = []

        system_message = (
            "You are an expert research scientist specializing in "
            "biomedical research. Write a single realistic, high-quality "
            "scientific abstract for a hypothetical bioRxiv research paper. "
            "Use a formal academic tone. Output only the abstract, "
            "without a title or introductory explanation."
        )

        for _, row in metadata_df.iterrows():

            if api_type == "random":
                user_message = (
                    "Generate a new synthetic research paper abstract "
                    "for the bioRxiv preprint server."
                )

            else:
                user_message = (
                    "Here is an example abstract:\n\n"
                    f"{row['text']}\n\n"
                    "Generate a new, unique abstract for a different "
                    "but related study. Keep a similar scientific "
                    "style and topic, but do not copy sentences."
                )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(prompt)

        return prompts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(df, iteration, elapsed):
    actual = df["text"].fillna("").astype(str).apply(
        lambda x: len(x.split())
    )

    result = {
        "iteration": iteration,
        "num_samples": len(df),
        "voting_mean": float(df["voting"].mean()),
        "voting_max": float(df["voting"].max()),
        "voting_std": float(df["voting"].std(ddof=0)),
        "actual_words_mean": float(actual.mean()),
        "word_error_mean": None,
        "word_error_rate": None,
        "elapsed_seconds": round(elapsed, 2),
    }

    if "word_count" in df.columns:
        target = pd.to_numeric(
            df["word_count"], errors="coerce"
        )

        valid = target.notna() & (target > 0)

        if valid.any():
            error = (actual[valid] - target[valid]).abs()

            result["word_error_mean"] = float(error.mean())
            result["word_error_rate"] = float(
                (error / target[valid]).mean()
            )

    return result



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--PE_T", type=int, default=1)
    parser.add_argument("--PE_num_syn", type=int, default=4)
    parser.add_argument("--PE_L_random", type=int, default=1)
    parser.add_argument("--num_private", type=int, default=100)
    parser.add_argument("--num_incontext", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_multiplier", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mini_experiment"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["maple", "augpe"],
        default="maple",
    )
    args = parser.parse_args()

    if args.PE_T < 0 or args.PE_num_syn < 1:
        parser.error("PE_T must be >= 0 and PE_num_syn >= 1")
    if args.PE_L_random < 1 or args.num_private < 1:
        parser.error("PE_L_random and num_private must be >= 1")
    if args.num_incontext < 1:
        parser.error("num_incontext must be >= 1")
    if args.noise_multiplier < 0:
        parser.error("noise_multiplier must be >= 0")

    set_random_seed(args.seed)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n=== Loading official MAPLE data ===")

    aim_files = list(DATA.glob("*aim*eps4.0*.csv"))
    if len(aim_files) != 1:
        raise RuntimeError(f"Expected one eps4 AIM file: {aim_files}")

    aim_df = pd.read_csv(aim_files[0])
    priv_df = pd.read_csv(DATA / "biorxiv_train_metadata.csv")
    val_df = pd.read_csv(DATA / "biorxiv_valid_metadata.csv")

    aim_df = aim_df.dropna(subset=METADATA)
    priv_df = priv_df.dropna(subset=["text"])
    val_df = val_df.dropna(subset=METADATA + ["text"])

    priv_df = priv_df.sample(
        n=min(args.num_private, len(priv_df)),
        random_state=args.seed
    ).copy()
    priv_df["PE.LABEL_ID"] = 0

    val_df = val_df.head(args.num_incontext)

    examples = [
        (row[METADATA].to_dict(), str(row["text"]))
        for _, row in val_df.iterrows()
    ]

    # Number of initial candidates follows the MAPLE
    # expansion rule: num_syn * (L + 1).
    initial_size = args.PE_num_syn * (args.PE_L_random + 1)

    if args.method == "maple":
        initial_df = aim_df.sample(
            n=initial_size,
            replace=initial_size > len(aim_df),
            random_state=args.seed
        ).reset_index(drop=True)

    else:
        # AugPE does not condition generation on metadata.
        initial_df = pd.DataFrame({
            "text": [""] * initial_size
        })

    print("Private samples:", len(priv_df))
    print("Initial candidates:", len(initial_df))
    print("Target synthetic samples:", args.PE_num_syn)

    print("\n=== Loading models ===")

    api_class = API if args.method == "maple" else AugPEAPI

    api = api_class(
        model_id=args.model_id,
        sampling_params=SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        ),
        examples=examples if args.method == "maple" else [],
        n_sample=args.num_incontext if args.method == "maple" else 0,
        METADATA=METADATA if args.method == "maple" else []
    )

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    vh = Voting_histogram(
        priv_df=priv_df,
        embedding=EmbeddingAdapter(embedding_model),
        batch_size=8,
        noise_multiplier=args.noise_multiplier
    )

    metrics = []
    experiment_start = time.perf_counter()

    # Initialization: RANDOM API + Voting
    print("\n=== Iteration 0: Initialization ===")
    start = time.perf_counter()

    candidates = api.RANDOM_API(initial_df.copy())
    candidates = vh.get_voting_histogram(candidates)

    cur_df = get_next_iteration(
        candidates,
        num_samples=args.PE_num_syn,
        selection_mode="rank"
    )

    elapsed = time.perf_counter() - start

    cur_df.to_csv(
        output_dir / "PE_iter_0.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics.append(evaluate(cur_df, 0, elapsed))
    print(metrics[-1])

    # Private Evolution iterations
    for iteration in range(1, args.PE_T + 1):
        print(f"\n=== Iteration {iteration}: Evolution ===")
        start = time.perf_counter()

        var_df = pd.concat(
            [cur_df.drop(columns=["voting"], errors="ignore")]
            * args.PE_L_random,
            ignore_index=True
        )

        var_df = api.VARIATION_API(var_df)

        candidates = pd.concat(
            [
                cur_df.drop(columns=["voting"], errors="ignore"),
                var_df,
            ],
            ignore_index=True
        )

        candidates = vh.get_voting_histogram(candidates)

        cur_df = get_next_iteration(
            candidates,
            num_samples=args.PE_num_syn,
            selection_mode="rank"
        )

        elapsed = time.perf_counter() - start

        cur_df.to_csv(
            output_dir / f"PE_iter_{iteration}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        metrics.append(evaluate(cur_df, iteration, elapsed))
        print(metrics[-1])

    summary = pd.DataFrame(metrics)

    summary.to_csv(
        output_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    total = time.perf_counter() - experiment_start

    print("\n=== Experiment Completed ===")
    print(summary.to_string(index=False))
    print(f"\nTotal time: {total:.2f} seconds")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
