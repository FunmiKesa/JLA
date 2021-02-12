cd src

CUDA_VISIBLE_DEVICES='1' \
# python train.py mot \
# --exp_id 'optim_mot17_ft_seed' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/optim_mot17_ft_seed/model_last.pth' \
# --num_epochs '20' \
# --lr_step '15' \
# --gpus '1' \
# --data_cfg '../src/lib/cfg/mot20.json' \
# --batch_size 4 \
# --forecast \
# --use_embedding \
# --multiprocessing_distributed \
# --dist-url tcp://127.0.0.1:5565 \
# --dist-backend 'nccl' \
# --world-size '1' \
# --rank '0' \
# --num_workers '4' \
# --resume
python track.py mot \
--exp_id 'optim_mot17_ft_seed' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/optim_mot17_ft_seed/model_last.pth' \
--conf_thres 0.4 \
--val_mot20 \

python track.py mot \
--exp_id 'optim_mot17_ft_seed' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/optim_mot17_ft_seed/model_last.pth' \
--conf_thres '0.4' \
--val_mot17 \

cd ..