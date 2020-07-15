#!/usr/bin/env bash

#SBATCH --job-name=serene
#SBATCH --gres=gpu:3
#SBATCH --qos=gpu-medium
#SBATCH --chdir=/fs/clip-quiz/entilzha/code/fever
#SBATCH --output=/fs/www-users/entilzha/logs/%A.log
#SBATCH --error=/fs/www-users/entilzha/logs/%A.log
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=gpu
#SBATCH --exclude=materialgpu00

set -x
hostname
nvidia-smi
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate serene
export SLURM_LOG_FILE="/fs/clip-quiz/entilzha/logs/${SLURM_JOB_ID}.log"
export MODEL_CONFIG_FILE="$2"
pwd
# $1 is the serialization dir, $2 is the model config
srun cp -n data/wiki_proto.sqlite3 /dev/shm/
srun python serene/main.py train $1 $2