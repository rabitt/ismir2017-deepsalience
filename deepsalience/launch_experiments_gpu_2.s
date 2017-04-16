#!/bin/bash
#
#SBATCH --job-name=mf0_exper2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
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

echo "Running Experiment 7"
python multif0_exper7_3.py
echo "Running Experiment 8"
python multif0_exper8_3.py
echo "Running Experiment 9"
python multif0_exper9_3.py
echo "Running Experiment 10"
python multif0_exper10_3.py
echo "Running Experiment 11"
python multif0_exper11_3.py
echo "Running Experiment 12"
python multif0_exper12_3.py
echo "Running Experiment 13"
python multif0_exper13_3.py
