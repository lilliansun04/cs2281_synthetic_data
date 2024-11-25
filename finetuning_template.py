import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os

# Set output directories and other global parameters
OUTPUT_DIR = "fine_tuning_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CustomTrainer(Trainer):
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

def preprocess_data(filepath, tokenizer, label_col, extra_col=None):
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

def main(model_name, data_filepath, label_col, extra_col=None):
    # Load tokenizer and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = preprocess_data(data_filepath, tokenizer, label_col, extra_col)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        save_steps=10,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
    )

    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Use train as eval for demonstration
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
    trainer.train()

if __name__ == "__main__":
    import sys
    model_name = "bert-base-uncased"  # Change this to the desired model
    data_filepath = "sample_data.csv"  # Path to your dataset
    label_col = "label"  # Column name for labels in the dataset
    extra_col = "extra_input"  # Optional column name for additional inputs
    main(model_name, data_filepath, label_col, extra_col)
