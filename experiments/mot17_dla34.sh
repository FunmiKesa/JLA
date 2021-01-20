cd src
# python train.py mot --exp_id mot17_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17.json'

python train.py mot --exp_id mot17_dla34 --load_model '/home/funmi/Experiments/FairMOT/exp/mot/mot17_dla34/model_last.pth' --data_cfg '../src/lib/cfg/mot17.json' --num_epochs 60 --resume
cd ..