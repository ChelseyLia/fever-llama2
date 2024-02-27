#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=80000M
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --output=fever_%j.txt
#SBATCH --error=myjob_error_%j.txt

module load python3.9  # Example: Adjust as needed for your setup

# Activate a virtual environment if you use one
#source /path/to/your/venv/bin/activate  # Adjust the path as needed

# Run your Python script
python3 /home/s1862623/diss/fever/1.py