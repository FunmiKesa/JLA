cd src

python train.py mot --load_model /home/funmi/Experiments/FairMOT/models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 8 --gpus 0,1 --forecast --num_epochs 30 --arch rnnforecast_34 --exp_id mot16_rnnforecast34 

cd ..