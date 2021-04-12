#!/bin/bash

#SBATCH -p pearl # partition (queue)
#SBATCH --job-name=exp18
#SBATCH -n 1  # number of CPU cores
#SBATCH --gres=gpu:4
#SBATCH --mem 250G
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH --cpus-per-task=16

singularity exec \
        --nv -w \
        ../../transfer sh -c "cd .. &&  sh experiments/mix_ft_ch_jla_10_60_2.sh"