cd src
CUDA_VISIBLE_DEVICES='1,2' python train.py mot --exp_id mot20_ft_mix_dla34 --load_model ../exp/mot/mot17_rnnforecast34_embedding/model_last.pth --num_epochs 20 --lr_step '15' --data_cfg '../src/lib/cfg/mot20.json'
cd ..