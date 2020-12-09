cd src
python track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/mot20_dla34_forecast/model_5.pth --conf_thres 0.6 --val_mot20 --batch_size 12 --gpu 2,3 --exp_id mot20_dla34_forecast
cd ..