cd src

python /home/funmi/Experiments/FairMOT/src/track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/mot16_rnnforecast34/model_last.pth --conf_thres 0.6 --val_mot16 --batch_size 8 --gpu 0,1 --forecast --exp_id mot16_rnnforecast34 --arch rnnforecast_34

cd ..