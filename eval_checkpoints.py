import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import ast
from finetuning_summarization_no_eval import (
    preprocess_summarization_data,
)
from rapidfuzz import process, fuzz
import argparse
import json


def generate_summary(input_ids, model, tokenizer):
    input_ids = input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=300)
    text = tokenizer.batch_decode(outputs)
    return text


def process_batch(batch, model, tokenizer):
    batch["input_ids"] = batch["input_ids"].to("cuda")
    batch["generated_summary"] = generate_summary(batch["input_ids"], model, tokenizer)
    return batch


def evaluate(checkpoint_path, tokenizer, summarization_val_dataset, verbose=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, device_map="auto")
    summarization_outputs = summarization_val_dataset.map(
        lambda batch: process_batch(batch, model, tokenizer),
        batched=True,
        batch_size=16,
    )

    total_keyword_match_percentage = 0
    for output, keywords in tqdm(
        zip(summarization_outputs, summarization_val_dataset["gpt_keywords"])
    ):
        keyword_matches = 0
        pred_words = output["generated_summary"].lower().split()
        for keyword in keywords:
            best_match = process.extractOne(
                keyword.lower(), pred_words, scorer=fuzz.ratio, score_cutoff=80
            )
            if best_match is not None:
                keyword_matches += 1
        match_percentage = (keyword_matches / len(keywords)) * 100
        if verbose:
            print(f"Completion: {output['generated_summary'].replace(tokenizer.pad_token, '')}\nKeywords: {keywords}, Match percentage: {match_percentage}")
        total_keyword_match_percentage += match_percentage

    return total_keyword_match_percentage / len(summarization_outputs)


def main(
    checkpoint_path, dataset_path, model_name, dataset_prop, output_dir, verbose=False
):
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    summarization_val_dataset = preprocess_summarization_data(
        dataset_path, tokenizer, dataset_prop
    )
    for root, dirs, files in os.walk(checkpoint_path):
        dirs.sort(key=lambda x: int(x.split("-")[-1]))
        for dir_name in dirs:
            checkpoint = os.path.join(root, dir_name)
            print("Processing", checkpoint)
            pct = evaluate(checkpoint, tokenizer, summarization_val_dataset, verbose)
            with open(os.path.join(output_dir, "evaluation_results.csv"), "a") as f:
                f.write(f"{dir_name},{pct}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarization model evaluation")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model on HuggingFace"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="/n/netscratch/idreos_lab/Lab/emyang/synthetic-data/cs2281_synthetic_data/synthetic/summary_val.csv",
        help="Path to the validation dataset",
    )
    parser.add_argument(
        "--dataset_prop",
        type=float,
        default=1.0,
        help="Proportion of the dataset to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/idreos_lab/Lab/emyang/synthetic-data/cs2281_synthetic_data/eval_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, os.path.basename(args.checkpoint_path))
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
        
    main(
        args.checkpoint_path,
        args.val_data_path,
        args.model_name,
        args.dataset_prop,
        output_dir,
        args.verbose,
    )
