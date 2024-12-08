#!/bin/sh
#SBATCH --job-name=finetuning_summarization
#SBATCH --partition=seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH -t 0-23:59                                                          # Runtime in D-HH:MM
#SBATCH -o /n/home11/lilliansun/cs2281_synthetic_data/results/output_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home11/lilliansun/cs2281_synthetic_data/results/output_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=lilliansun@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --account hlakkaraju_lab                        # Change to your lab's cluster account

module load Mambaforge/22.11.1-fasrc01
# Activate a virtual environment with torch, transformers installed
conda activate jupyter_py3.11

## logging
pushd /n/home11/lilliansun/cs2281_synthetic_data

CODE_DIR=/n/home11/lilliansun/cs2281_synthetic_data
pushd $CODE_DIR

model_name="google/gemma-2-2b-it"
output_dir="results/summarization/"

# TODO: change the scratch_dir to your personal scratch directory
scratch_dir="/n/netscratch/hlakkaraju_lab/Lab/lilliansun/synthetic_data"
# TODO: change the filepaths to the correct ones
summarization_train_filepath="summarization_train.json"
summarization_val_filepath="summarization_val.json"
summarization_test_filepath="summarization_test.json"

echo "Running finetuning_summarization.py"
## Run the experiment
# arguments: model_name, output_dir, scratch_dir,
#            summarization_train_filepath, summarization_val_filepath, summarization_test_filepath, 
python -u $CODE_DIR/finetuning_summarization.py $model_name $output_dir $scratch_dir $summarization_train_filepath $summarization_val_filepath $summarization_test_filepath
echo "DONE"