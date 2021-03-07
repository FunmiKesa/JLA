#!/bin/bash

#SBATCH -p pearl # partition (queue)
#SBATCH --job-name=exp9
#SBATCH -n 1  # number of CPU cores
#SBATCH --gres=gpu:2
#SBATCH --mem 150G
#SBATCH -t 0-05:00 # time (D-HH:MM)
#SBATCH --cpus-per-task=8 

# test if embedding result is the same
singularity exec \
        --nv -w \
        ../../transfer sh -c "cd .. &&  sh experiments/mot17_half_jla_10_60_no_emb.sh"