#!/bin/bash

#SBATCH -p pearl # partition (queue)
#SBATCH --job-name=exp4
#SBATCH -n 1  # number of CPU cores
#SBATCH --gres=gpu:2
#SBATCH --mem 150G
#SBATCH -t 0-05:00 # time (D-HH:MM)
#SBATCH --cpus-per-task=8

singularity exec \
        --nv -w \
        ../../transfer sh -c "cd .. &&  sh experiments/mot17_ft_ch_jla_10_30.sh"