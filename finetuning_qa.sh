#!/bin/sh
#SBATCH --job-name=finetuning_qa
#SBATCH --partition=seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH -t 0-23:59                                                         # Runtime in D-HH:MM
#SBATCH -o /n/home11/lilliansun/cs2281_synthetic_data/results/output_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home11/lilliansun/cs2281_synthetic_data/results/output_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=lilliansun@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --account hlakkaraju_lab

module load Mambaforge/22.11.1-fasrc01
# Activate a virtual environment with torch, transformers installed
conda activate jupyter_py3.11

## logging
pushd /n/home11/lilliansun/cs2281_synthetic_data

CODE_DIR=/n/home11/lilliansun/cs2281_synthetic_data
pushd $CODE_DIR

echo "Running finetuning_qa.py"
## Run the experiment
# arguments: model_name, 
#            qa_train_filepath, qa_val_filepath, qa_test_filepath
# TODO: change the filepaths to the correct ones
python -u $CODE_DIR/finetuning_qa.py "google/gemma-2-2b-it" "qa_train.json" "qa_val.json" "qa_test.json"

echo "DONE"