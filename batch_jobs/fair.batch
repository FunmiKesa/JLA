#!/bin/bash

#SBATCH -p pearl # partition (queue)
#SBATCH --job-name=fair
#SBATCH -n 1 # number of CPU cores
#SBATCH --mem 512G 
#SBATCH --gres=gpu:4
#SBATCH -t 0-07:00 # time (D-HH:MM)
#Number of processes per node to launch (20 for CPU, 2 for GPU)

#This command to run your pytorch script
#You will want to replace this

#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
echo $MASTER

#Get a random unused port on this host(MASTER) between 2000 and 9999
#First line gets list of unused ports
#2nd line restricts between 2000 and 9999
#3rd line gets single random port from the list

MASTER_PORT=`comm -23 <(seq 49000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep '[0-9]{1,5}' | sort -u)| shuf | head -n 1`

echo $MPORT

singularity exec \
        --nv -w \
	    ../transfer sh -c "cd src && python train.py mot --exp_id debug --arch rnnforecast_34 --load_model ../models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot17.json --batch_size 16 --num_workers 8 --gpus 0,1 --num_epochs 60 --forecast --use_embedding --multiprocessing_distributed --rank 0 --dist-url tcp://$MASTER:$MASTER_PORT --dist-backend nccl --world-size 1"

# python train.py mot --exp_id optim_mot_17 --arch rnnforecast_34 --load_model ../models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot17.json --batch_size 24 --num_workers 12 --gpus 0,1,2,3 --num_epochs 60 --forecast --use_embedding --multiprocessing_distributed --rank 0 --dist-url tcp://mn2:49394 --dist-backend nccl --world-size 1 

# srun -n 1 --gres=gpu:8 -t 0-07:00:00 --mem 512G --pty /bin/bash

singularity exec         --nv -w     ../transfer sh -c "cd src && OMP_NUM_THREADS=1 python train.py mot --exp_id optim_mot_17 --arch rnnforecast_34 --load_model ../exp/mot/optim_mot_17/model_last.pth --data_cfg ../src/lib/cfg/mot17.json --batch_size 24 --num_workers 16 --gpus 0,1,2,3 --num_epochs 60 --forecast --use_embedding --multiprocessing_distributed --rank 0 --dist-url tcp://$MASTER:$MASTER_PORT --dist-backend nccl --world-size 1 --resume "