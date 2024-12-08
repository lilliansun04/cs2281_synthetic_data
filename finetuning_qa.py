import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os

# Set output directories and other global parameters
OUTPUT_DIR = "results/qa/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

'''
TODO:
- Include data loader and tokenize on the fly
- Implement Seq2Seq loss
- Tokenize the output summaries
- Use fuzzy matching to match the keywords

Assumptions about dataset structure:
- Q&A: columns with GPT summaries, and yes/no questions
    - Each summary has 5 yes/no questions, have 5 different rows with same summary, each with a different question
'''

label_col = "label"  # Column name for labels in the dataset
extra_col = "extra_input"  # Optional column name for additional inputs

class QATrainer(Trainer):
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

def preprocess_qa_data(filepath, tokenizer, label_col, extra_col=None):
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

def main(model_name, qa_train_filepath, qa_val_filepath, qa_test_filepath):
    # Load tokenizer and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_train_dataset = preprocess_qa_data(qa_train_filepath, tokenizer, label_col, extra_col)
    qa_val_dataset = preprocess_qa_data(qa_val_filepath, tokenizer, label_col, extra_col)
    qa_test_dataset = preprocess_qa_data(qa_test_filepath, tokenizer, label_col, extra_col)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Initialize and train the Q&A trainer

if __name__ == "__main__":
    import sys
    # Script uses instruction-tuned Gemma 2B model
    model_name = str(sys.argv[1])  
    # Q&A task synthetic data
    qa_train_filepath = str(sys.argv[2])
    qa_val_filepath = str(sys.argv[3])
    qa_test_filepath = str(sys.argv[4])

    main(model_name, qa_train_filepath, qa_val_filepath, qa_test_filepath)