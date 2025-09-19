"""Main script for differentially private synthesis of bioRxiv schema data via AIM."""

import argparse
import logging
import os
import domain
import generate
import pandas as pd
import numpy as np

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

"""Main function to load Openreview schema data, define the data domain,

and generate synthetic data based on it.
"""

parser = argparse.ArgumentParser(
    description="Generate synthetic Biorxiv schema data."
)
parser.add_argument("--eps", type=float, default=4.0)
parser.add_argument(
    "--rho",
    type=float,
    default=0.04177113722835202,
    help="Rho value for the synthetic data generation.",
)
parser.add_argument(
    "--pgm_iters",
    type=int,
    default=2000,
    help=(
      "Number of optimization iterations for the synthetic data generation."
    ),
)
parser.add_argument(
    "--num_gen",
    type=int,
    default=2000,
    help="Number of synthetic samples to generate.",
)
parser.add_argument(
    "--data_path",
    type=str,
)
parser.add_argument(
    "--save_path",
    type=str,
)
args = parser.parse_args()

# --- 1. Load the real dataset ---
data_path = os.path.join("", args.data_path)
logging.info(f"Loading data from {data_path}...")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: Data file not found at {data_path}")
    logging.info(f"Successfully loaded {len(df)} records.")

# --- 1.1 preprocessing the word_count part ---
df['word_count'] = np.clip(df['word_count'].to_numpy(), a_min=50.0, a_max=1200.0)

# --- 2. Define the data schema (attribute_domains) ---
logging.info("Defining data schema...")


attribute_domains = {
  "area": domain.CategoricalAttribute([
      'Deep Learning and representational learning',
      'Applications (eg, speech processing, computer vision, NLP)',
     'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)',
     'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)',
     'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)',
     'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)',
     'Unsupervised and Self-supervised learning',
     'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )',
     'General Machine Learning',
     'Theory (eg, control theory, learning theory, algorithmic game theory)',
     'Generative models',
     'Optimization (eg, convex and non-convex optimization)',
  ]),
  "recommendation": domain.CategoricalAttribute([
      '6: marginally above the acceptance threshold',
     '3: reject, not good enough',
     '5: marginally below the acceptance threshold',
     '8: accept, good paper',
     '1: strong reject'
  ]),
  "primary_strength": domain.CategoricalAttribute([
      'Novelty/Originality of Idea', 'Strong Empirical Performance', 'Significant Problem or Application',
        'High-Quality Presentation/Clarity', 'Solid Theoretical Guarantees', 'Good Reproducibility (Code/Data Provided)'
  ]),
  "primary_weakness": domain.CategoricalAttribute([
      'Insufficient/Flawed Experiments', 'Weak or Missing Baselines', 'Lack of Novelty',
        'Unclear/Poor Presentation', 'Flawed Theoretical Claims', 'Limited Scope or Impact', 'Ethical Concerns'
  ]),
  "keywords_group_1": domain.CategoricalAttribute([
      'Large Language Models (LLMs)',
        'Diffusion Models',
        'Foundation Models',
        'Prompting',
        'Text-to-Image Synthesis',
        'Transformers',
        'Self-Supervised Learning (SSL)',
        'Datasets & Benchmarks',
        'Data Augmentation', 'Other'
  ]),
    "keywords_group_2": domain.CategoricalAttribute([
      'Few-Shot & Zero-Shot Learning',
        'Meta-Learning',
        'Continual Learning',
        'Transfer Learning',
        'Federated Learning',
        'Graph Neural Networks (GNNs)',
        'Knowledge Graphs',
        'Weak Supervision',
        'Causal Inference', 'Other'
  ]),
  "keywords_group_3": domain.CategoricalAttribute([
      'Efficiency',
        'Stochastic Gradient Descent (SGD)',
        'Model Compression',
        'Robustness',
        'Out-of-Distribution (OOD) Generalization',
        'Fairness & Bias',
        'Privacy',
        'Interpretability & Explainability',
        'Reinforcement Learning (RL)',
        'Decision Making', 'Other'
  ]),
  "word_count": domain.CategoricalAttribute(np.arange(50.0,1200.0,50)),
}

# --- 3. Run the synthetic data generation ---
logging.info("Starting synthetic data generation...")
t_start = pd.Timestamp.now()

synth = generate.run(
    attribute_domains,
    df,
    epsilon=0.0,
    delta=0.0,
    rho=args.rho,
    num_records=len(df),
    pgm_iters=1000,
)
logging.info(
  "Synthetic data generation completed in: %s", pd.Timestamp.now() - t_start
)

# --- 4. Save results ---
synth.to_csv(args.save_path,index=False)
