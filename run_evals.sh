#!/bin/sh
#SBATCH --job-name=summarization_eval
#SBATCH --partition=seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH -t 0-23:59                                                          # Runtime in D-HH:MM
#SBATCH -o /n/netscratch/idreos_lab/Lab/emyang/synthetic-data/cs2281_synthetic_data/eval_results/output_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/netscratch/idreos_lab/Lab/emyang/synthetic-data/cs2281_synthetic_data/eval_results/output_%j.err        # File to which STDERR will be written, %j inserts jobid

conda activate /n/home02/emyang/.conda/envs/model_collapse_20240911

export TASK="qna"
export CHECKPOINT_PATH="/n/netscratch/hlakkaraju_lab/Everyone/lilliansun/synthetic_data/qna/t5-large_1_no_eval_human"
export MODEL_NAME="google/flan-t5-large"
export OUTPUT_DIR="/n/netscratch/idreos_lab/Lab/emyang/synthetic-data/cs2281_synthetic_data/eval_results/qna"
export DATASET_PATH="/n/netscratch/idreos_lab/Lab/emyang/synthetic-data/cs2281_synthetic_data/synthetic/qna_val.csv"
export DATASET_PROP="0.25"
export VERBOSE="--verbose"

python eval_checkpoints.py --task $TASK --checkpoint_path $CHECKPOINT_PATH --model_name $MODEL_NAME --val_data_path $DATASET_PATH --dataset_prop $DATASET_PROP --output_dir $OUTPUT_DIR