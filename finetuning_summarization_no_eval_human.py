import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import Dataset
# from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import ast
import random

'''
Assumptions about dataset structure:
- Summarization: columns with raw texts, GPT summaries, and list of keywords
- Q&A: columns with GPT summaries, and yes/no questions
    - Each summary has 5 yes/no questions, have 5 different rows with same summary, each with a different question
'''

input_col = "article"
label_col = "human_summary"  # Use Human summaries as labels for benchmark
extra_col = "gpt_keywords"  # Optional column name for additional inputs

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

class SummarizationTrainer(Trainer):
    """
    Custom trainer that supports additional inputs and custom loss functions.
    """
    def __init__(self, args, model, train_dataset, extra_col):
        super().__init__(model, args = args, train_dataset = train_dataset)
        self.extra_col = extra_col

def preprocess_summarization_data(filepath, tokenizer, dataset_prop):
    """
    Preprocess the dataset, tokenize the text, and format for PyTorch.
    """
    df = pd.read_csv(filepath)
    df = df.iloc[:int(len(df) * dataset_prop)]

    input_articles = df[input_col].tolist()
    synthetic_summaries = df[label_col].tolist()
    keywords = [ast.literal_eval(item) if isinstance(item, str) else item for item in df[extra_col].tolist()]
    # Prefix the input text with the summarization task
    prefix = "Given the following text, summarize it in 3 to 5 sentences. "
    prefixed_articles = [prefix + article for article in input_articles]
    # Tokenize the input and output text
    input_tokenized = tokenizer(prefixed_articles, truncation=True, padding=True, max_length=3000)
    output_tokenized = tokenizer(synthetic_summaries, truncation=True, padding=True, max_length=256)
    dataset_dict = {
        "input_ids": input_tokenized["input_ids"],
        "attention_mask": input_tokenized["attention_mask"],
        "labels": output_tokenized["input_ids"],
    }
    if keywords:
        dataset_dict[extra_col] = keywords
    dataset_tokenized = Dataset.from_dict(dataset_dict)
    # Only set torch format for the tensor fields, not the keywords
    dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Size of input_ids tensor: {dataset_tokenized[0]['input_ids'].size()}, dtype: {dataset_tokenized[0]['input_ids'].dtype}")
    print(f"Actual sequence length: {dataset_tokenized[0]['attention_mask'].sum().item()}")

    return dataset_tokenized

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    
    # Debug prints
    print(f"Random test: {random.random()}")
    print(f"Numpy test: {np.random.rand()}")
    print(f"Torch test: {torch.rand(1).item()}")

def main(model_name, output_dir, scratch_dir, summarization_train_filepath, unique_save_name, dataset_prop, summarization_val_filepath, summarization_test_filepath, batch_size, eval_steps, random_seed):    # Load tokenizer and preprocess dataset
    set_all_seeds(random_seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarization_train_dataset = preprocess_summarization_data(summarization_train_filepath, tokenizer, dataset_prop)
    summarization_val_dataset = preprocess_summarization_data(summarization_val_filepath, tokenizer, dataset_prop)
    summarization_test_dataset = preprocess_summarization_data(summarization_test_filepath, tokenizer, dataset_prop)
    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Define training arguments
    summarization_training_args = TrainingArguments(
        output_dir=scratch_dir+unique_save_name,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        evaluation_strategy="no",  # Disables evaluation
        # evaluation_strategy="steps",
        logging_steps=eval_steps,
        # eval_steps=eval_steps,
        remove_unused_columns=False, # to keep extra columns during loss calculation
        # load_best_model_at_end=True,
        save_steps=eval_steps,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        seed=random_seed,
        # save_total_limit=2 # limit number of checkpoints for data storage
    )

    # Initialize summarization trainer
    summarization_trainer = SummarizationTrainer(
        model=model,
        args=summarization_training_args,
        train_dataset=summarization_train_dataset,
        # eval_dataset=summarization_val_dataset,
        extra_col=extra_col,
        # data_collator = data_collator,
    )

    f = open(output_dir+unique_save_name+".csv", "w")
    # INSERT CODE HERE to record metrics throughout training and evaluation

    # TODO: make sure to record metrics from eval before any training
    # summarization_trainer.evaluate()

    # Train the model
    summarization_trainer.train()

    # save the training history to a new csv file
    pd.DataFrame(summarization_trainer.state.log_history).to_csv(output_dir+unique_save_name+"_train.csv", header=True, index=False)

    # save results in csv file in output_dir


if __name__ == "__main__":
    import sys
    # Script uses instruction-tuned Gemma 2B model (or smaller T5 model for debugging)
    model_name = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    scratch_dir = str(sys.argv[3])
    unique_save_name = str(sys.argv[4])
    dataset_prop = float(sys.argv[5])

    # Summarization task synthetic data
    summarization_train_filepath = str(sys.argv[6])
    summarization_val_filepath = str(sys.argv[7])
    summarization_test_filepath = str(sys.argv[8])

    batch_size = int(sys.argv[9])
    eval_steps = int(sys.argv[10])
    random_seed = int(sys.argv[11])

    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Scratch directory: {scratch_dir}")
    print(f"Unique save name: {unique_save_name}")
    print(f"Dataset proportion: {dataset_prop}")
    print(f"Summarization train filepath: {summarization_train_filepath}")
    print(f"Summarization val filepath: {summarization_val_filepath}")
    print(f"Summarization test filepath: {summarization_test_filepath}")
    print(f"Batch size: {batch_size}")
    print(f"Evaluation steps: {eval_steps}")
    print(f"Random seed: {random_seed}")

    main(model_name, output_dir, scratch_dir, summarization_train_filepath, unique_save_name, dataset_prop, summarization_val_filepath, summarization_test_filepath, batch_size, eval_steps, random_seed)