import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, TrainerCallback
from datasets import Dataset
# from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import ast
from rapidfuzz import process, fuzz
import csv
import gc

'''
Assumptions about dataset structure:
- Summarization: columns with raw texts, GPT summaries, and list of keywords
- Q&A: columns with GPT summaries, and yes/no questions
    - Each summary has 5 yes/no questions, have 5 different rows with same summary, each with a different question
'''

input_col = "article"
label_col = "gpt_summary"  # Column name for labels in the dataset
extra_col = "gpt_keywords"  # Optional column name for additional inputs

class SummarizationTrainer(Trainer):
    """
    Custom trainer that supports additional inputs and custom loss functions.
    """
    def __init__(self, args, model, train_dataset, eval_dataset, extra_col):
        super().__init__(model, args = args, train_dataset = train_dataset, eval_dataset = eval_dataset)
        self.extra_col = extra_col
        # self.data_collator = data_collator

class MetricsLoggingCallback(TrainerCallback):
    """
    Custom callback to log metrics during training
    """
    def __init__(self, csv_writer, file_handle, training_args):
        self.writer = csv_writer
        self.file = file_handle
        self.training_args = training_args
        self.file.flush()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Called after evaluation
        """
        if state.is_world_process_zero:
            self.writer.writerow([
                state.global_step,
                metrics.get('eval_loss', None),
                metrics.get('eval_avg_keyword_match', None)
            ])
            self.file.flush()
            print(f"\nStep {state.global_step}:")
            print(f"Eval Loss: {metrics.get('eval_loss', None):.4f}")
            print(f"Avg Keyword Match: {metrics.get('eval_avg_keyword_match', None):.2f}%")
            torch.cuda.empty_cache()  # Clear cache after each eval
            gc.collect()


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

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def main(model_name, output_dir, scratch_dir, summarization_train_filepath, unique_save_name, dataset_prop, summarization_val_filepath, summarization_test_filepath, train_batch_size, eval_batch_size, eval_steps):
    # Load tokenizer and preprocess dataset
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
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="steps",
        logging_steps=eval_steps,
        eval_steps=eval_steps,
        remove_unused_columns=False, # to keep extra columns during loss calculation
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        # save_total_limit=2, # limit number of checkpoints for data storage
        save_steps=eval_steps # save 5 checkpoints throughout training, since approximately 100 evals per epoch
    )

    # def data_collator(features: list) -> dict:
    #     """
    #     Custom data collator for summarization with additional attribute handling.
    #     """
    #     batch = {key: [item[key] for item in features] for key in features[0].keys()}
    #     model_inputs = {
    #         "input_ids": pad_sequence(
    #             [torch.tensor(ids) for ids in batch["input_ids"]], 
    #             batch_first=True, 
    #             padding_value=tokenizer.pad_token_id
    #         ),
    #         "attention_mask": pad_sequence(
    #             [torch.tensor(ids) for ids in batch["attention_mask"]], 
    #             batch_first=True, 
    #             padding_value=0
    #         ),
    #         "labels": pad_sequence(
    #             [torch.tensor(ids) for ids in batch["labels"]], 
    #             batch_first=True, 
    #             padding_value=tokenizer.pad_token_id
    #         )
    #     }
    #     # Keep the keywords as a regular list, but store in a metadata field
    #     if extra_col in batch:
    #         model_inputs["_metadata_" + extra_col] = batch[extra_col]
    #     return model_inputs

    # Initialize summarization trainer
    summarization_trainer = SummarizationTrainer(
        model=model,
        args=summarization_training_args,
        train_dataset=summarization_train_dataset,
        eval_dataset=summarization_val_dataset,
        extra_col=extra_col,
        # data_collator = data_collator,
    )

    def compute_metrics(eval_preds):
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids
        # Convert logits to token ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=-1)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rest of your function remains the same
        keywords_list = summarization_trainer.eval_dataset[extra_col]
        metrics = {}
        total_keyword_match_percentage = 0

        for i in range(len(decoded_preds)):
            pred = decoded_preds[i]
            label = decoded_labels[i]
            keywords = keywords_list[i]
            keyword_matches = 0
            pred_words = pred.lower().split()
            for keyword in keywords:
                best_match = process.extractOne(
                    keyword.lower(), 
                    pred_words,
                    scorer=fuzz.ratio,
                    score_cutoff=80
                )
                if best_match is not None:
                    keyword_matches += 1
            match_percentage = (keyword_matches / len(keywords)) * 100
            # metrics["keyword_matches"].append(match_percentage)
            total_keyword_match_percentage += match_percentage
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"Keywords: {keywords}")
                print(f"Generated Summary: {pred}")
                print(f"Original Summary: {label}")
                print(f"Keyword Match Percentage: {match_percentage:.2f}%")  

        metrics["avg_keyword_match"] = total_keyword_match_percentage / len(decoded_preds)
        torch.cuda.empty_cache()
        return metrics

    summarization_trainer.compute_metrics = compute_metrics

    f = open(output_dir+unique_save_name+".csv", "w")
    writer = csv.writer(f)
    writer.writerow([
        'step',
        'eval_loss', 
        'eval_keyword_match_percentage'
    ])

    print("\nBenchmark Evaluation (Before Training):")
    benchmark_metrics = summarization_trainer.evaluate()
    writer.writerow([
        0,  # step 0 before training
        benchmark_metrics['eval_loss'],
        benchmark_metrics['eval_avg_keyword_match']
    ])

    metrics_callback = MetricsLoggingCallback(
        csv_writer=writer,
        file_handle=f,  # Pass the file handle for flushing
        training_args=summarization_training_args
    )
    summarization_trainer.add_callback(metrics_callback)

    # Train the model
    summarization_trainer.train()

    print("\nBenchmark Evaluation (After Training):")
    benchmark_metrics = summarization_trainer.evaluate()
    writer.writerow([
        -1,  # -1 to signify evaluation after training
        benchmark_metrics['eval_loss'],
        benchmark_metrics['eval_avg_keyword_match']
    ])
    f.flush()

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

    train_batch_size = int(sys.argv[9])
    eval_batch_size = int(sys.argv[10])
    eval_steps = int(sys.argv[11])

    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Scratch directory: {scratch_dir}")
    print(f"Unique save name: {unique_save_name}")
    print(f"Dataset proportion: {dataset_prop}")
    print(f"Summarization train filepath: {summarization_train_filepath}")
    print(f"Summarization val filepath: {summarization_val_filepath}")
    print(f"Summarization test filepath: {summarization_test_filepath}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Evaluation batch size: {eval_batch_size}")
    print(f"Evaluation steps: {eval_steps}")

    main(model_name, output_dir, scratch_dir, summarization_train_filepath, unique_save_name, dataset_prop, summarization_val_filepath, summarization_test_filepath, train_batch_size, eval_batch_size, eval_steps)