#!/bin/bash

#SBATCH -p pearl # partition (queue)
#SBATCH --job-name=exp15
#SBATCH -n 1  # number of CPU cores
#SBATCH --gres=gpu:2
#SBATCH --mem 150G
#SBATCH -t 0-05:00 # time (D-HH:MM)
#SBATCH --cpus-per-task=8 

singularity exec \
        --nv -B /mnt/beegfs/home/pearl022/funmi/FairMOT:/FairMOT\
                ../exp/conda sh -c ". /usr/local/etc/profile.d/conda.sh && conda activate FairMOT && cd /FairMOT && sh experiments/mot17_half_jla_10_60.sh"