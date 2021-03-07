cd src
CUDA_VISIBLE_DEVICES='0,1' python train.py mot \
--exp_id 'mot20_ft_jla34_10_60' \
--arch 'rnnforecast_34' \
--load_model '../jla_models/jla.pth' \
--data_cfg '../src/lib/cfg/mot20.json' \
--batch_size '16' \
--num_epochs '20' \
--lr_step '15' \
--num_workers '8' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--forecast \
--use_embedding \
--save_all
# --multiprocessing_distributed \
# --dist-url tcp://127.0.0.1:5564 \
# --dist-backend 'nccl' \
# --world-size '1' \
# --rank '0' \


python track.py mot --load_model ../exp/mot/mot20_ft_jla34_10_60/model_20.pth --conf_thres 0.3 --val_mot20 --exp_id mot20_ft_jla34_10_60 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60


python track.py mot --load_model ../exp/mot/mot17_ft_ch_jla_10_60/model_30.pth --conf_thres 0.4 --val_mot17 --exp_id mot17_ft_ch_jla_10_60 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60