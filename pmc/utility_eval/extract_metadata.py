# This script is designed to run in a Google Colab environment.
# It extracts structured metadata from a machine learning peer review
# using the Gemma-3-27b-it model via the Google Generative AI API.

import google.generativeai as genai
import json
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
import argparse

from pydantic import BaseModel
from typing import Literal
import textwrap
import typing



def annotate_features_prompt(
    text: str,
) -> str:
    """Returns a prompt to be used for feature extraction from text.

    This function generates a system prompt for an LLM to annotate a given text
    based on a predefined set of features. The features are derived from a
    Pydantic dataclass. The prompt instructs the LLM to output a JSON object
    containing the annotated features.

    Args:
    dataset_description: A 1-2 sentence description of the dataset.
    dataclass: A Pydantic dataclass defining the features to be extracted.
      Each field in the dataclass represents a feature, and the field's
      type annotation should specify the possible values (e.g., using
      typing.Literal).
    text: The text to be annotated.

    Returns:
    A detailed system prompt suitable for use with frontier language models
    for feature extraction, instructing the model to return a JSON object.
    """
    return textwrap.dedent("""
You are an expert clinical information extraction assistant. Your task is to carefully read a patient summary from a medical case report and extract the specified features according to the schema provided.

Output exactly one JSON object with no extra text or explanations.

**CRITICAL INSTRUCTION 1:** For any field in the JSON schema that lists specific options (e.g., "<Option1|Option2|...>"), you MUST select one of the provided options exactly as it is written. Do not invent, alter, or combine options. Failure to use an exact option from the list will be considered an error.

**CRITICAL INSTRUCTION 2:** Ensure the value chosen for a field is appropriate for that field's specific definition. Do not use an option from one field (e.g., 'Acute' from condition_chronicity) as the value for another field.

Use this schema:
```json
{{
  "medical_specialty": "<Cardiology|Dermatology|Dentistry & Oral Surgery|Endocrinology|Gastroenterology|Hematology|Infectious Disease|Nephrology|Neurology|Obstetrics & Gynecology|Oncology|Ophthalmology|Orthopedics|Otolaryngology (ENT)|Pediatrics|Psychiatry|Pulmonology|Rheumatology|Surgery|Urology|Other>", // The primary medical discipline. Map sub-specialties (e.g., Neurosurgery) to their primary field.
  "clinical_focus": "<Diagnostic Evaluation|Therapeutic Intervention|Monitoring & Follow-up|Adverse Event>", // The narrative's primary purpose (e.g., diagnosis, intervention, monitoring).
  "patient_age_group": "<Neonate (0-28 days)|Pediatric (29 days-12 years)|Adolescent (13-17 years)|Adult (18-64 years)|Geriatric (65+ years)>", // The patient's specific age category.
  "condition_chronicity": "<Acute|Chronic|Acute-on-Chronic|Recurrent / Relapsing|Congenital>", // The temporal nature and pattern of the patient's primary condition.
  "narrative_structure": "<Chronological History|Problem-Oriented Summary|Procedural Report|Other>", // The organizational style and flow of the clinical summary.
  "primary_intervention_type": "<Pharmacological|Surgical / Procedural|Supportive & Conservative Care|Not Applicable>", // The primary therapeutic or management action described (excluding diagnostic tests).
  "diagnostic_certainty": "<Confirmed Diagnosis|Provisional Diagnosis|Differential Diagnosis|Not Applicable>", // The level of diagnostic confidence expressed within the narrative.
  "patient_outcome": "<Resolved|Improved|Stable / Unchanged|Deteriorated|Deceased|Referred / Transferred|In-Progress / Unknown>" // The patient's clinical status or disposition at the end of the report.
}}
```

**Patient Summary to analyze:**

{text}

**Your output (JSON only):**
    """).format(
      text=text
    )


def extract_metadata_from_review(text: str):
    """
    Sends a peer review to the Gemma model and extracts structured metadata.

    Args:
        review_text: The full text of the peer review.

    Returns:
        A dictionary containing the extracted metadata, or None if an error occurs.
    """
    
    # Combine the system and user prompts. Gemma models perform well with this structure.
    full_prompt = annotate_features_prompt(text=text)

    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash-lite') # Using a capable model like Flash or Pro

        # Generate content
        print("Sending request to Gemma API...")
        response = model.generate_content(full_prompt)

        # Clean up the response text to ensure it's valid JSON
        # The model sometimes wraps the JSON in ```json ... ```
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()

        # print("\n--- Raw Model Response ---")
        # print(cleaned_text)
        # print("------------------------\n")

        # Parse the JSON response
        metadata = json.loads(cleaned_text)
        return metadata

    except Exception as e:
        print(f"An error occurred: {e}")
        # It's helpful to see the raw response text for debugging
        if 'response' in locals():
            print("\n--- Failed Raw Response Text ---")
            print(response.text)
            print("--------------------------------\n")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metadata for OpenReview dataset using Gemma API.")
    parser.add_argument("--load_path", help="Path to load data.")
    parser.add_argument("--save_path", help="Path to save annotated data.")
    parser.add_argument("--API_KEY", type=str, help="API_KEY for API call.")
    parser.add_argument("--max_output_tokens", type=int, default=512, help="MAX_OUTPUT_TOKENS")
    parser.add_argument("--temperature", type=float, default=1.0, help="TEMPERATURE")
    args = parser.parse_args()
    
    
    # --- Configuration ---
    #@title Set API key

    API_KEY = args.API_KEY 
    if API_KEY == "":
        raise ValueError("Please set your API key.")

    genai.configure(api_key=API_KEY)

    # Set a generation config:
    generation_config = genai.types.GenerationConfig(
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
    # Disable safey checkers, it also works if you remove this part
    safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    
    # Load data
    df = pd.read_csv(args.load_path)
    
    # A list containing multiple synthic text
    text_list = df['text'].tolist()
    
    all_metadata = []
    print(f"Found {len(text_list)} reviews to process.")

    for i, text in enumerate(text_list):
        print(f"\n--- Processing Text {i+1}/{len(text_list)} ---")
        extracted_data = extract_metadata_from_review(text)
        if extracted_data:
            all_metadata.append(extracted_data)
        else:
            # This means there are some errors, we append an empty dict to keep the shape consistency
            all_metadata.append({})
        # Add a small delay to avoid hitting API rate limits
        if i < len(text_list) - 1:
            print("Waiting for 1 second before the next API call...")
            time.sleep(0.02)
    
    if all_metadata:
        print("\n--- Final Result ---")
        # Convert the list of dictionaries to a pandas DataFrame. Remember to add "map_" prefix to column name.
        df_metadata = pd.DataFrame(all_metadata).add_prefix('map_')
        print("Successfully extracted metadata and created DataFrame:")
    
        # Join the new metadata columns to the original DataFrame
        result_df = pd.concat([df, df_metadata], axis=1)
        
        if 'Unnamed: 0' in result_df.keys():
            result_df = result_df.drop('Unnamed: 0',axis=1)
        
        result_df.to_csv(f'{args.save_path}', index=False)
        
    else:
        print("\nCould not extract metadata from any of the reviews.")
