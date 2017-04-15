#!/bin/bash
#
#SBATCH --job-name=multif0_exper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=50GB
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load ffmpeg/intel/3.2.2

source ~/.bashrc
unset XDG_RUNTIME_DIR

cd ~/repos/multif0/deepsalience

echo "Running Experiment 1"
python multif0_exper1.py
echo "Running Experiment 2"
python multif0_exper2.py
echo "Running Experiment 3"
python multif0_exper3.py
