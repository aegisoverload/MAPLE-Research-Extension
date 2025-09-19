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

"""Main function to load Biorxiv schema data, define the data domain,

and generate synthetic data based on it.
"""

parser = argparse.ArgumentParser(
    description="Generate synthetic Biorxiv schema data."
)
parser.add_argument("--eps", type=float, default=4.0)
parser.add_argument(
    "--rho",
    type=float,
    default=1000,
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
    default="../biorxiv/biorxiv_train_metadata.csv",
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
word_count = np.clip(df['word_count'].to_numpy(), a_min=50.0, a_max=650.0)
df['word_count'] = np.round(word_count / 50) * 50

# --- 2. Define the data schema (attribute_domains) ---
logging.info("Defining data schema...")

attribute_domains = {
  "primary_research_area": domain.CategoricalAttribute([
      "Biochemistry",
      "Bioinformatics",
      "Biophysics",
      "Cancer Biology",
      "Cell Biology",
      "Clinical Trials",
      "Developmental Biology",
      "Ecology",
      "Epidemiology",
      "Evolutionary Biology",
      "Genetics",
      "Genomics",
      "Immunology",
      "Microbiology",
      "Molecular Biology",
      "Neuroscience",
      "Paleontology",
      "Pathology",
      "Pharmacology and Toxicology",
      "Physiology",
      "Plant Biology",
      "Public Health",
      "Scientific Communication and Education",
      "Structural Biology",
      "Synthetic Biology",
      "Systems Biology",
      "Zoology",
      "Other",
  ]),
  "model_organism": domain.CategoricalAttribute([
      "Human",
      "Mouse/Rat",
      "Zebrafish",
      "Drosophila melanogaster",
      "Caenorhabditis elegans",
      "Saccharomyces cerevisiae",
      "Escherichia coli",
      "Arabidopsis thaliana",
      "Plant",
      "Cell Culture",
      "In Silico / Computational",
      "Other Mammal",
      "Other Vertebrate",
      "Other Invertebrate",
      "Other Microbe",
      "Not Applicable / Review",
      "Other",
  ]),
  "experimental_approach": domain.CategoricalAttribute([
      "Wet Lab Experimentation",
      "Computational / In Silico Analysis",
      "Clinical Study",
      "Field Study / Observation",
      "Case Study / Case Review",
      "Review / Meta-analysis",
      "New Method Development",
      "Theoretical Modeling",
      "Other",
  ]),
  "dominant_data_type": domain.CategoricalAttribute([
      "Genomic",
      "Transcriptomic",
      "Proteomic",
      "Metabolomic",
      "Imaging",
      "Structural",
      "Phenotypic / Behavioral",
      "Ecological / Environmental",
      "Clinical / Patient Data",
      "Simulation / Model Output",
      "Multi-omics",
      "Other",
  ]),
  "research_focus_scale": domain.CategoricalAttribute([
      "Molecular",
      "Cellular",
      "Circuit / Network",
      "Tissue / Organ",
      "Organismal",
      "Population",
      "Ecosystem",
      "Multi-scale",
      "Other",
  ]),
  "disease_mention": domain.CategoricalAttribute([
      "Cancer",
      "Neurodegenerative Disease",
      "Infectious Disease",
      "Metabolic Disease",
      "Cardiovascular Disease",
      "Autoimmune / Inflammatory Disease",
      "Psychiatric / Neurological Disorder",
      "Genetic Disorder",
      "No Specific Disease Mentioned",
      "Other",
  ]),
  "sample_size": domain.CategoricalAttribute([
      "Single Subject / Case Study",
      "Small Cohort (<50 subjects)",
      "Medium Cohort (50-1000 subjects)",
      "Large Cohort / Population-scale (>1000 subjects)",
      "Relies on Cell/Animal Replicates",
      "Not Specified / Not Applicable",
  ]),
  "research_goal": domain.CategoricalAttribute([
      "Investigating a mechanism",
      "Characterizing a system/molecule",
      "Developing a method/tool",
      "Identifying novel elements",
      "Testing a hypothesis",
      "Quantifying a parameter",
      "Evaluating/Comparing approaches",
      "Other",
  ]),
  "word_count": domain.CategoricalAttribute(np.arange(50.0,650.0,50)),
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
