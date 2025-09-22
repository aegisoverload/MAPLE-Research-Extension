import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
import argparse

def calculate_metrics(model, tokenizer, prompts, ground_truths, device='cpu', max_token_len=128, no_prompt=True):
    """
    Calculates perplexity and next-token prediction accuracy of a GPT-2 model.

    Perplexity is a measurement of how well a probability distribution or probability model
    predicts a sample. A low perplexity indicates the probability distribution is good at
    predicting the sample. It is calculated as the exponential of the cross-entropy loss.

    Accuracy is the percentage of tokens in the ground_truths that were correctly
    predicted by the model.

    Args:
        model (GPT2LMHeadModel): The pre-trained GPT-2 model.
        tokenizer (GPT2Tokenizer): The tokenizer for the model.
        prompts (list of str): A list of input prompts for the model.
        ground_truths (list of str): A list of corresponding ground truth texts.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
            - A list of perplexity scores for each individual pair.
            - The average perplexity score across all pairs.
            - A list of accuracy scores for each individual pair.
            - The average accuracy score across all pairs.
    """
    model.eval()
    perplexities = []
    accuracies = []
    total_neg_log_likelihood = 0
    total_tokens = 0
    total_correct_predictions = 0

    with torch.no_grad():
        for prompt, ground_truth in zip(prompts, ground_truths):
            if no_prompt:
                full_text = ground_truth
            else:
                # Concatenate prompt and ground truth to form the full text
                full_text = prompt + ground_truth
            
            # Tokenize the full text and move to the specified device
            inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=max_token_len)
            input_ids = inputs.input_ids.to(device)
            
            if no_prompt:
                prompt_token_length = 0
            else:
                # Determine the lengths of the prompt and ground truth in tokens
                prompt_token_length = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_token_len).input_ids.shape[1]
            ground_truth_token_length = input_ids.shape[1] - prompt_token_length

            # If there are no ground truth tokens, skip this pair
            if ground_truth_token_length == 0:
                perplexities.append(float('inf'))
                accuracies.append(0.0)
                continue

            # Create labels, masking the prompt tokens
            labels = input_ids.clone()
            labels[:, :prompt_token_length] = -100
            
            # Forward pass through the model
            outputs = model(input_ids, labels=labels)
            
            # --- Perplexity Calculation ---
            neg_log_likelihood = outputs.loss * ground_truth_token_length
            perplexity = torch.exp(outputs.loss)
            perplexities.append(perplexity.item())
            total_neg_log_likelihood += neg_log_likelihood.item()
            
            # --- Accuracy Calculation ---
            # Get the model's predictions (most likely token for each position)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Slice to get predictions and labels only for the ground_truth part
            # The logits at position i are predictions for token i+1.
            if no_prompt:
                ground_truth_predictions = predictions[:, prompt_token_length:-1]
                ground_truth_labels = input_ids[:, prompt_token_length+1:]
            else:
                ground_truth_predictions = predictions[:, prompt_token_length - 1:-1]
                ground_truth_labels = input_ids[:, prompt_token_length:]

            # Compare predictions to actual labels
            correct_predictions = (ground_truth_predictions == ground_truth_labels).sum().item()
            accuracy = correct_predictions / ground_truth_token_length
            accuracies.append(accuracy)

            # Update totals for calculating the average accuracy and perplexity
            total_correct_predictions += correct_predictions
            total_tokens += ground_truth_token_length

    # Calculate overall averages
    average_perplexity = torch.exp(torch.tensor(total_neg_log_likelihood / total_tokens)).item() if total_tokens > 0 else float('inf')
    average_accuracy = total_correct_predictions / total_tokens if total_tokens > 0 else 0.0
    
    return perplexities, average_perplexity, accuracies, average_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate NTP.")
    parser.add_argument("--target_dir", help="Path to the directory of saved model.")
    parser.add_argument("--tst_file_path", help="Path to test dataset file.")
    parser.add_argument("--save_path", help="Path to the saved result.")
    parser.add_argument("--device", type=str, default="cuda", help="GPU/CPU to compute.")
    parser.add_argument("--device", type=str, default="cuda", help="GPU/CPU to compute.")
    args = parser.parse_args()
    print(args)
    
    df_test = pd.read_csv(args.tst_file_path)
    
    # Create prompts column. This one should align with how you train the downstream model.
    def create_prompt(row):
        prompt_template = (
            # "<s>[INST] Generate a peer review for a machine learning research paper "
            # "with the following characteristics.\n\n"
            # "**Area:** {area}\n"
            # "**Recommendation:** {recommendation} [/INST]\n"
            " "
        )
        # return prompt_template.format(
        #     area=row['area'],
        #     recommendation=row['recommendation'],
        # )
        return prompt_template
    
    # Apply the function to create the new 'content' column
    df_test['prompt'] = df_test.apply(create_prompt, axis=1)
    prompts = df_test['prompt'].to_list()
    ground_truths = df_test['text'].to_list()
    
    max_token_len = 512 # This is for 'prajjwal1/bert-small'. Change it if you are evaluating with different model.
    model = AutoModelForCausalLM.from_pretrained(args.target_dir).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_dir)

    # Set pad token if it's not set. This is only for bert model.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    _, average_perplexity, _, average_accuracy = calculate_metrics(model, tokenizer, prompts, ground_truths, args.device, max_token_len, no_prompt=True)
    print("\n--- Results ---")
    print(f"Model dir:{args.target_dir}")
    print(f"  Average Perplexity: {average_perplexity:.2f}")
    print(f"  Average Accuracy:   {average_accuracy:.4f}")
    print("----------------")
    
    # save results
    np.save(args.save_path, np.array([average_perplexity,average_accuracy]))
