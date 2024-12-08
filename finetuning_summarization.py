import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os


'''
TODO:
- Write method to preprocess dataset, make sure to keep the 5 keywords and human_summary
    - Tokenize the text with prompting instructions
- Write compute_metrics method, check w/ old research script to make sure it is correctly assigned to trainer, and outputs every eval_steps
    - Use fuzzy matching to match the keywords
- Implement Seq2Seq loss (this is just default loss for AutoModelForSeq2SeqLM)

Assumptions about dataset structure:
- Summarization: columns with raw texts, GPT summaries, and list of keywords
- Q&A: columns with GPT summaries, and yes/no questions
    - Each summary has 5 yes/no questions, have 5 different rows with same summary, each with a different question
'''

label_col = "label"  # Column name for labels in the dataset
extra_col = "extra_input"  # Optional column name for additional inputs

class SummarizationTrainer(Trainer):
    """
    Custom trainer that supports additional inputs and custom loss functions.
    """
    def __init__(self, *args, extra_input_key=None, custom_loss_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_input_key = extra_input_key
        self.custom_loss_func = custom_loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract additional input if specified
        extra_input = inputs.pop(self.extra_input_key, None) if self.extra_input_key else None
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Compute custom loss if provided, otherwise default to CrossEntropyLoss
        if self.custom_loss_func:
            loss = self.custom_loss_func(logits, labels, extra_input)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

def preprocess_summarization_data(filepath, tokenizer):
    """
    Preprocess the dataset, tokenize the text, and format for PyTorch.
    """
    df = pd.read_csv(filepath)
    texts = df["text"].tolist()
    labels = df[label_col].tolist()
    extra_inputs = df[extra_col].tolist() if extra_col else None

    tokenized = tokenizer(texts, truncation=True, padding=True, max_length=128)
    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }
    if extra_inputs:
        dataset_dict[extra_col] = extra_inputs

    return Dataset.from_dict(dataset_dict)

def custom_loss_function(logits, labels, extra_input=None):
    """
    Example custom loss function that incorporates extra inputs.
    """
    loss = torch.nn.functional.cross_entropy(logits, labels)
    if extra_input is not None:
        # Example regularization term based on extra_input
        reg_term = torch.mean(extra_input.float()) * 0.01
        loss += reg_term
    return loss

def main(model_name, output_dir, scratch_dir, summarization_train_filepath, summarization_val_filepath, summarization_test_filepath):
    # Load tokenizer and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarization_train_dataset = preprocess_summarization_data(summarization_train_filepath, tokenizer)
    summarization_val_dataset = preprocess_summarization_data(summarization_val_filepath, tokenizer)
    summarization_test_dataset = preprocess_summarization_data(summarization_test_filepath, tokenizer)
    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Define training arguments
    summarization_training_args = TrainingArguments(
        output_dir=scratch_dir,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        save_steps=60,
        logging_steps=10,
        eval_steps=10,
        remove_unused_columns=False, # to keep extra columns during loss calculation
        load_best_model_at_end=True,
        save_total_limit=2 # limit number of checkpoints for data storage
    )

    # Initialize summarization trainer
    summarization_trainer = SummarizationTrainer(
        model=model,
        args=summarization_training_args,
        train_dataset=summarization_train_dataset,
        eval_dataset=summarization_test_dataset,  # Use train as eval for demonstration
        data_collator=lambda data: {
            "input_ids": pad_sequence([torch.tensor(f["input_ids"]) for f in data], batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": pad_sequence([torch.tensor(f["attention_mask"]) for f in data], batch_first=True, padding_value=0),
            "labels": torch.tensor([f["labels"] for f in data]),
            extra_col: torch.tensor([f[extra_col] for f in data]) if extra_col else None,
        },
        extra_input_key=extra_col,
        custom_loss_func=custom_loss_function,
    )

    # Train the model
    summarization_trainer.train()


if __name__ == "__main__":
    import sys
    # Script uses instruction-tuned Gemma 2B model
    model_name = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    scratch_dir = str(sys.argv[3])
     
    # Summarization task synthetic data
    summarization_train_filepath = str(sys.argv[4])
    summarization_val_filepath = str(sys.argv[5])
    summarization_test_filepath = str(sys.argv[6])

    main(model_name, output_dir, scratch_dir, summarization_train_filepath, summarization_val_filepath, summarization_test_filepath)