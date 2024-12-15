import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
# from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import ast

'''
Assumptions about dataset structure:
- Summarization: columns with raw texts, GPT summaries, and list of keywords
- Q&A: columns with GPT summaries, and yes/no questions
    - Each summary has 5 yes/no questions, have 5 different rows with same summary, each with a different question
'''

summary_col = "gpt_summary"
question_col = "Q"
label_col = "A"  # Column name for labels in the dataset

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def preprocess_qna_data(filepath, tokenizer, dataset_prop):
    """
    Preprocess the dataset, tokenize the text, and format for PyTorch.
    """
    df = pd.read_csv(filepath)
    # TODO: REMOVE subsetting dataset after debugging
    df = df.iloc[:int(len(df) * dataset_prop)]
    df = df.astype({label_col: int}) # ensure 0/1 are integers

    input_summaries = df[summary_col].tolist()
    input_questions = df[question_col].tolist()
    labels = df[label_col].tolist()
    # Prefix the input text with the Q&A task
    prefix = "Given the following summary, answer this question using 1 for yes and 0 for no. Question: "
    prefixed_questions = [prefix + input_questions[i] + " Summary: " + input_summaries[i] for i in range(len(input_summaries))]
    # Tokenize the input and output text
    input_tokenized = tokenizer(prefixed_questions, truncation=True, padding=True, max_length=512)
    dataset_dict = {
        "input_ids": input_tokenized["input_ids"],
        "attention_mask": input_tokenized["attention_mask"],
        "labels": labels,
    }
    dataset_tokenized = Dataset.from_dict(dataset_dict)
    # Only set torch format for the tensor fields, not the keywords
    dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Size of input_ids tensor: {dataset_tokenized[0]['input_ids'].size()}, dtype: {dataset_tokenized[0]['input_ids'].dtype}")
    print(f"Actual sequence length: {dataset_tokenized[0]['attention_mask'].sum().item()}")

    return dataset_tokenized


def main(model_name, output_dir, scratch_dir, qna_train_filepath, unique_save_name, dataset_prop, qna_val_filepath, qna_test_filepath, batch_size, eval_steps):    # Load tokenizer and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qna_train_dataset = preprocess_qna_data(qna_train_filepath, tokenizer, dataset_prop)
    qna_val_dataset = preprocess_qna_data(qna_val_filepath, tokenizer, dataset_prop)
    qna_test_dataset = preprocess_qna_data(qna_test_filepath, tokenizer, dataset_prop)
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Define training arguments
    qna_training_args = TrainingArguments(
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
        # save_total_limit=2 # limit number of checkpoints for data storage
    )

    # Initialize Q&A trainer
    qna_trainer = Trainer(
        model=model,
        args=qna_training_args,
        train_dataset=qna_train_dataset,
    )

    # Train the model
    qna_trainer.train()

    # save the training history to a new csv file
    pd.DataFrame(qna_trainer.state.log_history).to_csv(output_dir+unique_save_name+"_train.csv", header=True, index=False)

    # save results in csv file in output_dir


if __name__ == "__main__":
    import sys
    # Script uses instruction-tuned Gemma 2B model (or smaller T5 model for debugging)
    model_name = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    scratch_dir = str(sys.argv[3])
    unique_save_name = str(sys.argv[4])
    dataset_prop = float(sys.argv[5])

    # Q&A task synthetic data
    qna_train_filepath = str(sys.argv[6])
    qna_val_filepath = str(sys.argv[7])
    qna_test_filepath = str(sys.argv[8])

    batch_size = int(sys.argv[9])
    eval_steps = int(sys.argv[10])

    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Scratch directory: {scratch_dir}")
    print(f"Unique save name: {unique_save_name}")
    print(f"Dataset proportion: {dataset_prop}")
    print(f"Q&A train filepath: {qna_train_filepath}")
    print(f"Q&A val filepath: {qna_val_filepath}")
    print(f"Q&A test filepath: {qna_test_filepath}")
    print(f"Batch size: {batch_size}")
    print(f"Evaluation steps: {eval_steps}")

    main(model_name, output_dir, scratch_dir, qna_train_filepath, unique_save_name, dataset_prop, qna_val_filepath, qna_test_filepath, batch_size, eval_steps)