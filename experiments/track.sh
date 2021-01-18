cd src
# python track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/rnnforecast_34_16/model_last.pth --conf_thres 0.6 --val_mot15 --batch_size 12 --gpu 2,3 --exp_id rnnforecast_34_16

# python train.py mot --load_model /home/funmi/Experiments/FairMOT/models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 12 --gpus 0,1 --forecast --num_epochs 30 --arch rnnforecast_34 --exp_id rnnforecast_34_mask

python train.py mot --load_model /home/funmi/Experiments/FairMOT/models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 12 --gpus 0,1 --forecast --num_epochs 30 --arch rnnforecast_34 --exp_id rnnforecast_34_latest2
cd ..