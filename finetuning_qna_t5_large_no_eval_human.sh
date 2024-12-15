#!/bin/sh
#SBATCH --job-name=finetuning_qna_full
#SBATCH --partition=kempner_h100 #kempner #seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH -t 0-23:59                                                          # Runtime in D-HH:MM
#SBATCH -o /n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data/results/output_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data/results/output_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=lilliansun@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --account kempner_sham_lab #hlakkaraju_lab                        # Change to your lab's cluster account

module load Mambaforge/22.11.1-fasrc01
# Activate a virtual environment with torch, transformers installed
conda activate jupyter_py3.11

## logging
pushd /n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data

CODE_DIR=/n/holylabs/LABS/hlakkaraju_lab/Users/lilliansun/cs2281_synthetic_data
pushd $CODE_DIR

# Start with smaller 80M model
model_name="google/flan-t5-large"
# TODO: increase model size once code is fully debugged
# model_name="google/gemma-2-2b-it"
output_dir="results/qna/"

# TODO: change the scratch_dir to your personal scratch directory
scratch_dir="/n/netscratch/hlakkaraju_lab/Everyone/lilliansun/synthetic_data/qna/"
dataset_prop=1 # how much of the dataset to use (as a fraction not a percentage)
# TODO: give a unique name for saving results **IMPORTANT: giving same name as a previous run will overwrite the results and checkpoints!**
# TODO: edit unique_save_name to match model name
unique_save_name="t5-large_"$dataset_prop"_no_eval_human_3"
qna_train_filepath="synthetic/qna_train.csv"
qna_val_filepath="synthetic/qna_val.csv"
qna_test_filepath="synthetic/qna_test.csv"

batch_size=8
eval_steps=40

echo "Running finetuning_qna_no_eval_human.py"
echo "Saving results to $output_dir$unique_save_name"
## Run the experiment
# arguments: model_name, output_dir, scratch_dir,
#            qna_train_filepath, qna_val_filepath, qna_test_filepath, 
python -u $CODE_DIR/finetuning_qna_no_eval_human.py $model_name $output_dir $scratch_dir $unique_save_name $dataset_prop $qna_train_filepath $qna_val_filepath $qna_test_filepath $batch_size $eval_steps
echo "DONE"