#!/bin/sh
#SBATCH --job-name=summarization_eval
#SBATCH --partition=kempner_h100 #kempner #seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH -t 0-23:59                                                          # Runtime in D-HH:MM
#SBATCH -o /n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data/results/output_%j.out         # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data/results/output_%j.err        # File to which STDERR will be written, %j inserts jobid
#  #SBATCH --array=1-3
#SBATCH --account kempner_sham_lab #hlakkaraju_lab                        # Change to your lab's cluster account
#SBATCH --mail-user=lilliansun@college.harvard.edu

# conda activate /n/home02/emyang/.conda/envs/model_collapse_20240911
module load Mambaforge/22.11.1-fasrc01
# Activate a virtual environment with torch, transformers installed
conda activate jupyter_py3.11

export TASK="summarize"
export CHECKPOINT_PATH="/n/netscratch/hlakkaraju_lab/Everyone/lilliansun/synthetic_data/t5-large_1_no_eval_seed_"$SLURM_ARRAY_TASK_ID
export MODEL_NAME="google/flan-t5-large"
export OUTPUT_DIR="/n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data/eval_results/summarize"
export DATASET_PATH="/n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data/synthetic/summary_val.csv"
export DATASET_PROP="0.25"
export VERBOSE="--verbose"

python eval_checkpoints.py --task $TASK --checkpoint_path $CHECKPOINT_PATH --model_name $MODEL_NAME --val_data_path $DATASET_PATH --dataset_prop $DATASET_PROP --output_dir $OUTPUT_DIR  --baseline
echo DONE