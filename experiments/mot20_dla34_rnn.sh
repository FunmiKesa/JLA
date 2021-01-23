cd src
CUDA_VISIBLE_DEVICES='0,1' python train.py mot --exp_id forecast_rnn_16 --load_model ../models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 12 --gpus 0,1 --forecast --num_epochs 60
cd ..