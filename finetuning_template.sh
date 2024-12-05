#!/bin/sh
#SBATCH --job-name=finetuning_template
#SBATCH --partition=seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40gb
#SBATCH -t 0-02:00                                                          # Runtime in D-HH:MM
#SBATCH -o /n/home11/lilliansun/cs2281_synthetic_data/results/output_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home11/lilliansun/cs2281_synthetic_data/results/output_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=lilliansun@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --account hlakkaraju_lab

module load Mambaforge/22.11.1-fasrc01
conda activate jupyter_py3.11

## logging
pushd /n/home11/lilliansun/cs2281_synthetic_data

CODE_DIR=/n/home11/lilliansun/cs2281_synthetic_data
pushd $CODE_DIR

echo "Running finetuning_template.py"
## Run the experiment
python -u $CODE_DIR/finetuning_template.py

echo "DONE"