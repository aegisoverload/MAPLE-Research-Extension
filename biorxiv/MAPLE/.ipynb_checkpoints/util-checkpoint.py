# These three only for fixing random seed.
import torch
import os
import random
import re
# This is for finding nearest in-context examples.
import heapq

###
import pandas as pd
import numpy as np
from vllm.v1.sample.logits_processor import LogitsProcessor
from transformers import AutoTokenizer

def set_seed(seed_value=42):
    """
    Sets the seed for all relevant random number generators to ensure reproducibility.

    Args:
        seed_value (int): The integer value to use as the seed.
    """
    # 1. Set the seed for Python's built-in random module
    random.seed(seed_value)

    # 2. Set the seed for NumPy
    np.random.seed(seed_value)

    # 3. Set the seed for PyTorch
    torch.manual_seed(seed_value)

    # 4. Set the seed for PyTorch's CUDA operations, if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU setups
    
    # This should not affect the results too much... I decide to prioritize the speed.
    # # 5. Configure PyTorch to use deterministic algorithms
    # # Note: This can have a performance impact, so it's a trade-off
    # # between reproducibility and speed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 6. Set Python's hash seed
    # This is important for operations that rely on hash randomization.
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    print(f"All random seeds have been set to: {seed_value}")
    
def get_top_n_by_hamming_distance(examples, new_metadata, n):
    """
    Extracts the top n examples with the smallest Hamming distance to a new metadata dictionary.

    The Hamming distance between two dictionaries with the same keys is defined as
    the number of keys where their corresponding values are different.

    Args:
        examples (list): A list of tuples, where each tuple is
                         (example_metadata, example_text).
        new_metadata (dict): The reference metadata dictionary to compare against.
        n (int): The number of top examples to return.

    Returns:
        list: A list containing the top n examples (tuples) with the smallest
              Hamming distance.
    """
    # Ensure n is not larger than the number of available examples
    n = min(n, len(examples))

    # Define a helper function to calculate Hamming distance for the key function
    def calculate_hamming_distance(example):
        example_metadata = example[0]
        # Count the number of keys where the values are different
        distance = sum(1 for key in new_metadata if new_metadata[key] != example_metadata[key])
        return distance

    # Use heapq.nsmallest to efficiently find the top n items
    top_n_examples = heapq.nsmallest(n, examples, key=calculate_hamming_distance)

    return top_n_examples

def sample_metadata_for_randomapi(df_trn, target_ids, num_sample_per_id, PE_L_random):
    temp_list = []
    for target_id in target_ids:
        # This step now should be done carefully. We will randomly sample args.PE_L+1 rows from cur_df[cur_df['PE.LABEL_ID']==target_id]
        # Note that metadata are not identical for the same PE.LABEL_ID
        candidate = df_trn[df_trn['PE.LABEL_ID']==target_id]
        num_sample_rand = PE_L_random*num_sample_per_id[target_id]
        num_full = num_sample_rand//len(candidate)
        num_remain = num_sample_rand - num_full*len(candidate)
        for _ in range(num_full):
            temp_list.append(candidate)
        # sampling num_remain 
        if num_remain>0:
            # Ideally, we use sampling without replacement to encourage diveristy.
            temp_list.append(candidate.sample(n=num_remain,replace=False))

    return pd.concat(temp_list, ignore_index=True)

# Define a custom logits processor to suppress Chinese tokens
class SuppressChineseLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config, device, is_pin_memory
    ):
        self.req_info: dict[int, SamplingParams] = {}
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_device = device
        # The logic to find Chinese tokens remains the same
        self.chinese_token_ids = self._get_chinese_token_ids()

    def _get_chinese_token_ids(self):
        chinese_token_ids = []
        chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
        for token_id in range(self.tokenizer.vocab_size):
            token_str = self.tokenizer.decode(token_id, skip_special_tokens=True)
            if chinese_char_pattern.search(token_str):
                chinese_token_ids.append(token_id)
        print(f"Found and suppressing {len(chinese_token_ids)} Chinese tokens.")
        return torch.tensor(chinese_token_ids, device=self.llm_device, dtype=torch.long)

    # --- NEW REQUIRED METHODS ---

    def apply(self, logits, sampling_metadata=None):
        """
        This is the main method that replaces the old `__call__`.
        It applies the logic to the logits.
        """
        logits[:, self.chinese_token_ids] = -float('inf')
        return logits

    def is_argmax_invariant(self) -> bool:
        """
        This method tells vLLm if your processor can change the highest-probability token.
        Since we might suppress the current highest token, we return False.
        """
        return False

    def update_state(self, sampling_metadata=None):
        """
        This method is for stateful processors that need to update internal state.
        Ours is stateless, so we don't need to do anything here.
        """
        pass