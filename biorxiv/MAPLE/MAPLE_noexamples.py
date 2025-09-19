import os
import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor
from transformers import AutoTokenizer
import argparse
import numpy as np
import json
import re
import torch 
import heapq
import random

from pe.histogram.nearest_neighbor_backend.sklearn import search
from pe.embedding.text import SentenceTransformer

from util import set_seed, get_top_n_by_hamming_distance, sample_metadata_for_randomapi, SuppressChineseLogitsProcessor

def create_review_generation_prompt(new_metadata, examples = None, n_sample=5):
    """
    Generates a structured prompt for a chat-based LLM to create a synthetic
    academic peer review using few-shot examples.

    Args:
        new_metadata (dict): The metadata for the review to be generated.
        examples (list): A list of tuples, where each tuple contains
                         (metadata_dict, review_text) for a few-shot example.

    Returns:
        list: A list of dictionaries representing the conversation, ready for
              a chat model API.
    """
    
    # 1. System Message: Sets the persona and overall goal for the model.
    system_message = {
        "role": "system",
        "content": (
            "You are an expert research scientist specializing in biomedical research. Your task is to write a single, realistic, and high-quality abstract for a new research paper intended for the bioRxiv preprint server. The abstract must follow the standard scientific structure: Background, Methods, Results, and Conclusion. The tone must be formal and academic, using appropriate technical terminology. CRITICAL RULE: Your entire response must consist ONLY of the abstract's text. Do not include a title, author names, keywords, or any introductory phrases like 'Here is the abstract:'."
        )
    }
    
    # 2. First User Message (MODIFIED):
    # We combine the main instructions and the first few-shot example into ONE user turn
    # to fix the user/user sequence error.
    instructions = f"""
**Task:** Your goal is to generate a single, high-quality, and realistic synthetic paper abstract for the bioRxiv preprint server **only** based on the provided metadata.

**Instructions:**

1.  **Invent from Specification:** You must write a completely new paper abstract from scratch. The abstract's content, arguments, and structure must be a logical and detailed embodiment of the provided **[METADATA]**.
2.  **Justify the Metadata:** Every aspect of your paper abstract must support the metadata.
3.  **Adhere to Length:** The generated review should be approximately the length specified by the `word_count` in the metadata (deviate no more than 25 words). \n
"""
    
    new_metadata_str = json.dumps(new_metadata, indent=2)
    instructions += f"""
Now, following all the instructions, generate a new synthetic paper abstract for the metadata below. Remember, your response must ONLY be the text of the paper abstract.
**[METADATA]:**
```json
{new_metadata_str}
```
"""
    messages = [system_message ,{"role": "user", "content": instructions.strip()}]
    
    return messages


def create_review_variation_prompt(new_metadata, text_to_variate, examples):
    """
    Generates a structured prompt for a chat-based LLM to variate a given
    academic peer review.

    Args:
        new_metadata (dict): The metadata for the review.
        review_to_variate (str): The original review text to be variated.
        examples (list): A list of tuples, where each tuple contains
                         (metadata_dict, review_text) to demonstrate style.

    Returns:
        list: A list of dictionaries representing the conversation, ready for
              a chat model API.
    """
    
    messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert research scientist specializing in biomedical research. Your task is to write a single, novel, and realistic abstract for a hypothetical research paper. This new abstract must be thematically and stylistically similar to an example abstract provided by the user. Emulate its topic, technical language, structure, and tone. \n\n**CRITICAL RULES:**\n1. **DO NOT COPY:** You must create a new abstract describing a different, hypothetical study. Do not simply paraphrase or copy sentences from the provided example. \n2. **OUTPUT FORMAT:** Your entire response must consist ONLY of the new abstract's text. Do not include any introductory phrases, titles, or other text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the example abstract:\n\n{text_to_variate}\n\nNow, generate a new, unique abstract for a different but related study. Make sure the resulting abstract has approximately {int(new_metadata['word_count'])} word."
                )
            }
        ]
    
    
    return messages


class API:
    def __init__(self, model_id, sampling_params, examples, n_sample=5, METADATA=None):
        """
        model_id: can be name from huggingface or local path to your pretrain model directory. In the later case remember to put tokenizer there as well.
        """
        # This is mainly for prompt construction.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if 'Qwen' in model_id:
            llm_device = 'cuda:0'
            # Initialize LLM and its tokenizer.
            self.llm = LLM(model=model_id, max_model_len=32768, logits_processors=[SuppressChineseLogitsProcessor]) 
        else:
            # Initialize LLM and its tokenizer.
            self.llm = LLM(model=model_id, max_model_len=32768) 
        # This is default sampling params.
        self.sampling_params = sampling_params
        
        # examples
        self.examples = examples
        self.n_sample = n_sample
        
        # metadata
        self.METADATA = METADATA
        
    def _get_api_prompts(self, metadata_df, api_type='random'):
        """
        template: a template to be filled for the prompts.
        metadata_df: a dataframe containing metadata to be filled in the random_api_template.
        We use len(metadata_df) to define the number of prompts. So even if no metadata, one should create a dummy one (i.e., with only 'text' column containing empty string).
        api_type: 'random' or 'variation'.
        
        Return:
        list of prompts
        """
        assert api_type in ['random','variation']
        prompts = []
        
        # --- Iterate through the DataFrame and generate each prompt ---
        for index, row in metadata_df.iterrows():
            try:
                if api_type == 'random':
                    messages = create_review_generation_prompt(row[self.METADATA].to_dict(), self.examples, self.n_sample)
                else:
                    messages = create_review_variation_prompt(row[self.METADATA].to_dict(),  row['text'], self.examples)
                    
                new_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(new_prompt)

            except KeyError as e:
                print(f"Warning: Skipping row {index} because a key is missing from the DataFrame that is present in the template: {e}")
            except Exception as e:
                print(f"An unexpected error occurred at row {index}: {e}")
            

        return prompts
    
    def RANDOM_API(self, metadata_df, sampling_params=None):
        
        if sampling_params is None:
            # use default sampling params
            sampling_params = self.sampling_params
        
        # Get prompts for RANDOM_API
        prompts = self._get_api_prompts(metadata_df,api_type='random')
        
        # Generate
        outputs = self.llm.generate(prompts, sampling_params)
        metadata_df["text"] = [output.outputs[0].text for output in outputs]
        
        return metadata_df
    
    def VARIATION_API(self, var_df, sampling_params=None, L=1):
        """
        df: the dataframe containing current text at column "text", and potentially other metadata
            
        
        Return:
        df with variated texts in 'nxt_text' column.
        """
        
        if sampling_params is None:
            # use default sampling params
            sampling_params = self.sampling_params
        
        # Get prompts for VARIATION_API
        prompts = self._get_api_prompts(var_df,api_type='variation')
        outputs = self.llm.generate(prompts, sampling_params)
        var_df["text"] = [output.outputs[0].text for output in outputs]
        
        return var_df

class Voting_histogram:
    def __init__(self, priv_df, embedding, batch_size=100, num_nearest_neighbors=1, mode='L2', histogram_threshold=None, noise_multiplier=None):
        self.embedding = embedding
        self.batch_size = batch_size
        self._num_nearest_neighbors = num_nearest_neighbors # Warning, current implementation only support neighbor=1.
        self._mode = mode
        
        # DP NN part
        self.histogram_threshold = histogram_threshold
        self.noise_multiplier = noise_multiplier
        
        if 'embedding' in priv_df.columns:
            print('Use precomputed embedding...')
            self._priv_emb = np.stack(priv_df['embedding'].apply(lambda s: np.fromstring(
        s.replace('\n', '').replace('[', '').replace(']', ''), sep=' '))) # This might need to be adjust depending on how the embedding are saved...
        else:
            # When initialize, compute and store the private embedding.
            self._priv_emb = self.get_emb(priv_df)

        # We also create a dictionary for PE.LABEL_ID to index mapping
        id_list = priv_df['PE.LABEL_ID'].to_list()
        id_dict = {}
        for index, i in enumerate(id_list):
            if i in id_dict:
                id_dict[i].append(index)
            else:
                id_dict[i] = [index]
                
        self._priv_id_dict = id_dict
        
        
    def get_emb(self,df):
        """
        df: a dataframe with column 'text' containing text to be embeded.
        
        Return:
        emb: a np array of shape (#row of df, emb dim).
        """
        return self.embedding._model.encode(df['text'].tolist(), batch_size=self.batch_size)
    
    def get_num_sample_per_target_id(self, num_sample):
        """
        Get the number of target sample per target_id.
        """
        id_list = list(self._priv_id_dict.keys())
        
        # First compute the fraction
        count_list = []
        for ID in id_list:
            id_num = len(self._priv_id_dict[ID])
            count_list.append(id_num)
            
        count_list = np.array(count_list)
        id_ratio = count_list / np.sum(count_list)
        target_num_per_id = id_ratio * num_sample
        num_per_id = np.floor(target_num_per_id).astype(int)
        num_per_id_left = num_sample - np.sum(num_per_id)
        ids = np.argsort(target_num_per_id - num_per_id)[::-1]
        num_per_id[ids[:num_per_id_left]] += 1
        assert np.sum(num_per_id) == num_sample
        if np.any(num_per_id == 0):
            raise ValueError("num_samples is so small that the number of samples for some label ids is zero.")
            
        # Return a dictionary to map id to num_per_id
        return dict(zip(id_list, num_per_id))
    
    def get_voting_histogram(self, sub_cur_df, target_id=None):
        """
        sub_cur_df: a dataframe with column 'text' containing text to be embeded. This is the current synthetic data. These rows corresponds to 'PE.LABEL_ID'=target_id.
        target_id: 'PE.LABEL_ID'=target_id.
        
        Return:
        cur_df: with the voting histogram stored in the column `voting`. 
        """
        
        sub_cur_emb = self.embedding._model.encode(sub_cur_df['text'].tolist(), batch_size=self.batch_size)
        
        if isinstance(target_id,int):
            sub_priv_emb = self._priv_emb[self._priv_id_dict[target_id]]
        else:
            sub_priv_emb = self._priv_emb

        
        # NN voting:
        _, ids = search(
            syn_embedding=sub_cur_emb,
            priv_embedding=sub_priv_emb,
            num_nearest_neighbors=self._num_nearest_neighbors,
            mode=self._mode,
        )
        ids = ids.flatten()
        # Get sub histogram based on original ids and add it to the overall count.
        histogram = np.bincount(ids, minlength=len(sub_cur_df)).astype(np.float32)
        
        if self.noise_multiplier:
            histogram += np.random.normal(scale=self.noise_multiplier, size=len(sub_cur_df))
            print("Add DP noise...")
        
        # post-processing histogram
        if not (self.histogram_threshold is None):
            histogram = np.clip(histogram, a_min=self.histogram_threshold, a_max=None)
            histogram -= self.histogram_threshold
            print("Do thresholding...")
            
        # Store the final histogram
        sub_cur_df['voting'] = histogram
        return sub_cur_df 

def get_next_iteration(sub_cur_df, num_samples, selection_mode="rank"):
    """
    sub_cur_df: a dataframe with column 'voting' containing voting histogram. This is the current synthetic data.
    num_samples: number of samples we want to extract.
    selection_mode: do sampling or simply extract data based on ranking (with repetition to keep the dataset size consistent).
    
    Return: cur_df, with optionally the 'voting' column removed.
    """
    assert selection_mode in ['sample','rank']
    
    if selection_mode == 'sample':
        histogram = sub_cur_df['voting']
        prob = histogram / histogram.sum()
        # This is critical to prevent np random choice error.
        prob = np.asarray(prob).astype('float64')
        prob = prob / np.sum(prob)
        indices = np.random.choice(len(sub_cur_df), size=num_samples, p=prob, replace=True)
        new_df = sub_cur_df.iloc[indices]
    else:
        histogram = sub_cur_df['voting']
        indices = np.argsort(histogram)[::-1][:num_samples]
        new_df = sub_cur_df.iloc[indices]
    
    new_df = new_df.reset_index(drop=True) # This prevents unexpected indexing error.
    return new_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta PE with vllm acceleration.")
    parser.add_argument("--target_dir", help="Path to the directory of saved model.")
    parser.add_argument("--priv_file_path", help="Path to private dataset file with metadata.")
    parser.add_argument("--train_file_path", help="Path to training dataset (aim) file with metadata.")
    parser.add_argument("--val_file_path", help="Path to validation dataset file with metadata.")
    parser.add_argument("--emb_name", type=str, default="sentence-t5-base", help="Emb model name.")
    parser.add_argument("--emb_dim", type=int, default=1024, help="Emb dimension.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM that provides API call. Should use instruction following models.")
    parser.add_argument("--num_incontext", type=int, default=50, help="Number of in-context examples.")
    parser.add_argument("--n_sample", type=int, default=5, help="Number selected in-context examples.")
    parser.add_argument("--temperature", type=float, default=1.0, help="vllm sampling params")
    parser.add_argument("--top_p", type=float, default=0.95, help="vllm sampling params")
    parser.add_argument("--max_tokens", type=int, default=1024, help="vllm sampling params")
    parser.add_argument("--rng_seed", type=int, default=42, help="fixing random seed for reproducibility.")
    parser.add_argument("--PE_T", type=int, default=2, help="How many PE shots.")
    parser.add_argument("--PE_L_random", type=int, default=6, help="number of random_api call per iteration in PE")
    parser.add_argument("--PE_variate", action='store_true', help="Whether to polish text with Variation API.")
    parser.add_argument("--PE_num_syn", type=int, default=2000, help="number of targeted synthetic data for PE")
    parser.add_argument("--PE_continue_iter", type=int, default=-1, help="continue from a prior saved results")
    parser.add_argument("--PE_metalist", type=str, default=None, help="List of int(will be converted in code). Will only use METADATA[PE_metalist] as metadata. Default to use all, must contain word_count. Sep by ,.")
    parser.add_argument("--PE_noise_multiplier", type=float, default=0, help="DP noise multiplier")
    args = parser.parse_args()
    print(args)

    # Set random seed
    set_seed(args.rng_seed)
    # make directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)
    
    model_id = args.model_id
    # Remember to change this for different dataset
    METADATA = ["primary_research_area","model_organism","experimental_approach","dominant_data_type","research_focus_scale","disease_mention","sample_size","research_goal","word_count"]
    if args.PE_metalist:
        meta_list = np.array([int(item) for item in args.PE_metalist.split(',')]).astype(int)
        # Otherwise we use all metadata.
        METADATA = [METADATA[i] for i in meta_list]
    print(f'Used metadata features: {METADATA}')
    
    # --- Prepare the examples list from the DataFrame using 1-3 validation data ---
    in_context_examples = []
    example_indices = np.arange(args.num_incontext) # We use this to reduce the randomness over different runs... Can be further optimized
    df_val = pd.read_csv(args.val_file_path)
    for index in example_indices:
        row = df_val.loc[index]
        review_text = row['text']
        # Create metadata dict by dropping the review text column
        metadata_dict = row[METADATA].to_dict()
        in_context_examples.append((metadata_dict, review_text))


    ### === Initialize API ===
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens = args.max_tokens)
    api = API(model_id, sampling_params, in_context_examples, args.n_sample, METADATA)
    
    ### === Prepare initial private and synthetic df ===
    # === Load training data ===
    df_trn = pd.read_csv(args.train_file_path)
    df_trn['PE.LABEL_ID'] = 0
    prv_df = pd.read_csv(args.priv_file_path)
    prv_df['PE.LABEL_ID'] = 0
    # We don't need PE.LABEL_ID for biorxiv, but this will be used for the other dataset so we keep it for consistency.
    
    
    df_trn = df_trn[['PE.LABEL_ID']+METADATA] # keep only metadata (non-private/DPfied part)
    num_syn = args.PE_num_syn
    
    ### === Initialize voting module ===
    embedding = SentenceTransformer(model=args.emb_name)
    # This can be changed. For our manuscript we use sentence-t5 embedding by default.
    if "sentence-t5" in args.emb_name:
        embedding._model.max_seq_length = args.emb_dim
        # One could change this to cpu if there only 1 available GPU...
        embedding._model = embedding._model.to('cuda:1')
        
    # Initializing VH module
    # Remember to set histogram_threshold >= 0.0 if you will use selection_mode = sample.
    VH = Voting_histogram(prv_df, embedding, histogram_threshold=None, noise_multiplier=args.PE_noise_multiplier)
    target_ids = list(set(prv_df['PE.LABEL_ID'].to_list()))
    num_sample_per_id = VH.get_num_sample_per_target_id(num_syn)
    
    ### === PE starts here ===
    if args.PE_continue_iter < 0:
        # Initialization of PE with RANDOM_API. Note that we will use num_syn*(PE_L_random+1) as the number of seed.
        print("Initializing PE with RANDOM_API")
        cur_df = sample_metadata_for_randomapi(df_trn, target_ids, num_sample_per_id, args.PE_L_random+1)
        cur_df = api.RANDOM_API(cur_df)
        if args.PE_variate:
            cur_df = api.VARIATION_API(cur_df)

        # Voting
        cur_df = VH.get_voting_histogram(cur_df)
        # Sample next generation
        cur_df = get_next_iteration(cur_df, num_samples=num_sample_per_id[0], selection_mode='rank')

        # Save result
        cur_df.to_csv(f'{args.target_dir}/PE_iter_0.csv', index=False)
        print(f"PE iteration 0 ends and results saved.")
        # ===== rest PE iterations =====
        for iteration in range(1,args.PE_T+1):
            rnd_df = sample_metadata_for_randomapi(df_trn, target_ids, num_sample_per_id, 5)
            rnd_df = api.RANDOM_API(rnd_df)
            if args.PE_variate:
                rnd_df = api.VARIATION_API(rnd_df)
            var_df = pd.concat([cur_df] * 1, ignore_index=True)
            var_df = api.RANDOM_API(var_df)
            if args.PE_variate:
                var_df = api.VARIATION_API(var_df)
            cur_df = pd.concat([cur_df,rnd_df,var_df], ignore_index=True)
            
            # Voting
            cur_df = VH.get_voting_histogram(cur_df)
            # Sample next generation
            cur_df = get_next_iteration(cur_df, num_samples=num_sample_per_id[0], selection_mode='rank')
            
            # Save result
            cur_df.to_csv(f'{args.target_dir}/PE_iter_{iteration}.csv', index=False)
            print(f"PE iteration {iteration} ends and results saved.")
    else:
        print(f'Continue from {args.target_dir}/PE_iter_{args.PE_continue_iter}.csv')
        cur_df = pd.read_csv(f'{args.target_dir}/PE_iter_{args.PE_continue_iter}.csv')
        # ===== rest PE iterations =====
        for iteration in range(args.PE_continue_iter+1,args.PE_T+1):
            rnd_df = sample_metadata_for_randomapi(df_trn, target_ids, num_sample_per_id, 5)
            rnd_df = api.RANDOM_API(rnd_df)
            if args.PE_variate:
                rnd_df = api.VARIATION_API(rnd_df)
            var_df = pd.concat([cur_df] * 1, ignore_index=True)
            var_df = api.RANDOM_API(var_df)
            if args.PE_variate:
                var_df = api.VARIATION_API(var_df)
            cur_df = pd.concat([cur_df,rnd_df,var_df], ignore_index=True)
            
            # Voting
            cur_df = VH.get_voting_histogram(cur_df)
            # Sample next generation
            cur_df = get_next_iteration(cur_df, num_samples=num_sample_per_id[0], selection_mode='rank')
            
            # Save result
            cur_df.to_csv(f'{args.target_dir}/PE_iter_{iteration}.csv', index=False)
            print(f"PE iteration {iteration} ends and results saved.")
