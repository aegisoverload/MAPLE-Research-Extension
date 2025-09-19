# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series without DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import datasets
import dp_transformers
import transformers
import sys
import logging

from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig

import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

def get_syn_dataset(path='', valid_path=None):
    # Load the synthetic dataset
    df = pd.read_csv(path)
    # This df should have at least a column ['text']
    
    # Define a function to format each row into a prompt. We just used the suggested prompt from Gemini.
    def create_prompt(row):
        prompt_template = (
            "{text}"
        )
        return prompt_template.format(
            text=row['text']
        )
    
    # Apply the function to create the new 'content' column
    df['content'] = df.apply(create_prompt, axis=1)
    
    # Select only the final 'content' column
    df = df[['content']]
    
    if valid_path:
        # Use actual validation set.
        print('Use actual validation set')
        df_val = pd.read_csv(valid_path)
        df_val['content'] = df_val.apply(create_prompt,axis=1)
        df_val = df_val[['content']]
        D_train = Dataset.from_pandas(df)
        D_test = Dataset.from_pandas(df_val)
    else:
        print('Split validation set from train.')
        # We will randomly sample 10% rows as validation data for this part.
        # Do random shuffling
        df = df.sample(frac=1)
        # Convert the Pandas DataFrame to a Hugging Face Dataset
        # We name it as test for consistency of the original code. The actual test set will only be used when evaluating the generated data.
        D_train = Dataset.from_pandas(df[:int(len(df)*0.9)])
        D_test = Dataset.from_pandas(df[int(len(df)*0.9):])
    
    dataset = DatasetDict({
        'train': D_train,
        'test': D_test
    })
    return dataset

@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        return LoraConfig(**params)

@dataclass
class CustomArguments:
    """
    Custom arguments for our specific training script.
    """
    train_data_path: str = field(
        metadata={"help": "Path to the training dataset file."}
    )
    valid_data_path: str = field(default="",
        metadata={"help": "Path to the validation dataset file."}
    )


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    model: ModelArguments
    lora: LoraArguments
    custom: CustomArguments

                     

def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    # Load training (and optionally validation) dataset
    dataset = get_syn_dataset(path=args.custom.train_data_path, valid_path=args.custom.valid_data_path)


    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # For bert
    if 'bert' in args.model.model_name:
        tokenizer.pad_token = '[PAD]'
        model.config.pad_token_id = model.config.eos_token_id
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        # make sure we use masked attention for causal language
        if not model.config.is_decoder:
            model.config.is_decoder = True
    

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(batch['content'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )
    
    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    trainer = transformers.Trainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model()
    output_dir = trainer.args.output_dir
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, ModelArguments, LoraArguments, CustomArguments))
    train_args, model_args, lora_args, custom_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args, lora=lora_args, custom=custom_args))
