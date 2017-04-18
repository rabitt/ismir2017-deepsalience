#!/bin/bash
#
#SBATCH --job-name=mf0_exper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
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

echo "Running harmonic zscore Experiment 1"
python multif0_exper1_batchin.py
echo "Running harmonic zscore Experiment 2"
python multif0_exper2_batchin.py
echo "Running harmonic zscore Experiment 3"
python multif0_exper3_batchin.py
echo "Running harmonic zscore Experiment 4"
python multif0_exper4_batchin.py
echo "Running harmonic zscore Experiment 5"
python multif0_exper5_batchin.py
echo "Running harmonic zscore Experiment 6"
python multif0_exper6_batchin.py
echo "Running harmonic zscore Experiment 7"
python multif0_exper7_batchin.py
echo "Running harmonic zscore Experiment 8"
python multif0_exper8_batchin.py
echo "Running harmonic zscore Experiment 9"
python multif0_exper9_batchin.py
echo "Running harmonic zscore Experiment 10"
python multif0_exper10_batchin.py
echo "Running harmonic zscore Experiment 11"
python multif0_exper11_batchin.py
echo "Running harmonic zscore Experiment 12"
python multif0_exper12_batchin.py
echo "Running harmonic zscore Experiment 13"
python multif0_exper13_batchin.py
echo "Running harmonic zscore Experiment 14"
python multif0_exper14_batchin.py
