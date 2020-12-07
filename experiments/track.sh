cd src
python track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/mot20_dla34/model_15.pth --conf_thres 0.6 --val_mot20 --batch_size 12 --gpu 0,1
cd ..