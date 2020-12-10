cd src
CUDA_VISIBLE_DEVICES='2,3' python train.py mot --exp_id forecast_rnn --load_model ../models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot20.json --batch_size 8 --gpus 2,3 --forecast
cd ..