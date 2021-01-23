cd src

python train.py mot --load_model /home/funmi/Experiments/FairMOT/models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 8 --gpus 0,1 --forecast --num_epochs 30 --arch rnnforecast_34 --exp_id mot16_rnnforecast34_embedding --use_embedding

python /home/funmi/Experiments/FairMOT/src/track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/mot16_rnnforecast34_embedding/model_last.pth --conf_thres 0.6 --val_mot16 --batch_size 8 --gpu 0,1 --forecast --exp_id mot16_rnnforecast34_embedding --arch rnnforecast_34 --use_embedding

cd ..