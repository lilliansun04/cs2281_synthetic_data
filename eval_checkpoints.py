import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
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
from finetuning_qna_no_eval import preprocess_qna_data
from rapidfuzz import process, fuzz
import argparse
import json
from sklearn.metrics import accuracy_score

def generate_summary(input_ids, model, tokenizer):
    outputs = model.generate(input_ids, max_new_tokens=300)
    text = tokenizer.batch_decode(outputs)
    return text


def generate_answer(input_ids, attention_mask, model, tokenizer):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    label = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return label


def get_ngrams(text, n):
    """
    Generate n-grams from text.
    """
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def evaluate_summary(
    summarization_outputs, summarization_val_dataset, tokenizer, verbose
):
    """
    Evaluates summaries against keywords, supporting multi-word keywords.
    Returns the average keyword match percentage across all summaries.
    """
    total_keyword_match_percentage = 0
    
    for output, keywords in tqdm(
        zip(summarization_outputs, summarization_val_dataset["gpt_keywords"])
    ):
        keyword_matches = 0
        generated_text = output["generated_summary"].lower()
        
        # Generate n-grams up to the length of the longest keyword
        # max_keyword_length = max(len(keyword.split()) for keyword in keywords)
        all_ngrams = []
        for n in range(1, 6):
            all_ngrams.extend(get_ngrams(generated_text, n))
        
        # Match each keyword against the appropriate n-grams
        for keyword in keywords:
            keyword = keyword.lower()
            best_match = process.extractOne(
                keyword, all_ngrams, scorer=fuzz.ratio, score_cutoff=80
            )
            
            if best_match is not None:
                keyword_matches += 1
        
        match_percentage = (keyword_matches / len(keywords)) * 100
        
        if verbose:
            print(
                f"Completion: {output['generated_summary'].replace(tokenizer.pad_token, '')}\n"
                f"Keywords: {keywords}, Match percentage: {match_percentage}"
            )
            
        total_keyword_match_percentage += match_percentage

    # Calculate and return the average match percentage
    return total_keyword_match_percentage / len(summarization_outputs)

def evaluate_qna(qna_outputs, qna_val_dataset, tokenizer, verbose):
    accuracy = accuracy_score(qna_val_dataset["labels"], qna_outputs["prediction"])
    # if verbose:
    #     for output, label in zip(qna_outputs, qna_val_dataset["labels"]):
    #         print(
    #             f"Question: {tokenizer.decode(torch.tensor(qna_outputs['input_ids']))}, Prediction: {output['prediction']}, Label: {label}, Match: {output['prediction'] == label}"
    #         )
    return accuracy


def process_batch(task, batch, model, tokenizer, verbose=False):
    if task == "summarize":
        batch["input_ids"] = batch["input_ids"].to("cuda")
        batch["generated_summary"] = generate_summary(
            batch["input_ids"], model, tokenizer
        )
        return batch
    elif task == "qna":
        batch["input_ids"] = batch["input_ids"].to("cuda")
        batch["attention_mask"] = batch["attention_mask"].to("cuda")
        batch["prediction"] = generate_answer(
            batch["input_ids"], batch["attention_mask"], model, tokenizer
        )
        if verbose:
            print(
                f"Input: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}"
            )
            print(f"Prediction: {batch['prediction'][0]}")
        return batch


def evaluate(task, checkpoint_path, tokenizer, val_dataset, verbose=False):
    if task == "summary":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path, device_map="auto"
        )
        summarization_outputs = val_dataset.map(
            lambda batch: process_batch(task, batch, model, tokenizer),
            batched=True,
            batch_size=16,
        )

        return evaluate_summary(summarization_outputs, val_dataset, tokenizer, verbose)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, device_map="auto"
        )
        qna_outputs = val_dataset.map(
            lambda batch: process_batch(task, batch, model, tokenizer, verbose),
            batched=True,
            batch_size=16,
        )

        return evaluate_qna(qna_outputs, val_dataset, tokenizer, verbose)


def main(
    task,
    checkpoint_path,
    dataset_path,
    model_name,
    dataset_prop,
    output_dir,
    verbose=False,
):
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if task == "summarize":
        val_dataset = preprocess_summarization_data(
            dataset_path, tokenizer, dataset_prop
        )
    else:
        val_dataset = preprocess_qna_data(dataset_path, tokenizer, dataset_prop)
    for root, dirs, files in os.walk(checkpoint_path):
        dirs.sort(key=lambda x: int(x.split("-")[-1]))
        for dir_name in dirs:
            checkpoint = os.path.join(root, dir_name)
            print("Processing", checkpoint)
            pct = evaluate(task, checkpoint, tokenizer, val_dataset, verbose)
            with open(os.path.join(output_dir, "evaluation_results.csv"), "a") as f:
                f.write(f"{dir_name},{pct}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarization model evaluation")
    parser.add_argument(
        "--task",
        type=str,
        choices=["summarize", "qna"],
        required=True,
    )
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
        args.task,
        args.checkpoint_path,
        args.val_data_path,
        args.model_name,
        args.dataset_prop,
        output_dir,
        args.verbose,
    )
