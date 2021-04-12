#!/bin/bash

#SBATCH -p pearl # partition (queue)
#SBATCH --job-name=exp19
#SBATCH -n 1  # number of CPU cores
#SBATCH --gres=gpu:4
#SBATCH --mem 250G
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH --cpus-per-task=16

singularity exec \
        --nv -w \
        ../../transfer sh -c "cd .. &&  sh experiments/citywalks_ft_jla34_10_60.sh"