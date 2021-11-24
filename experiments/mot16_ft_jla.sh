cd src
CUDA_VISIBLE_DEVICES='0,1' python train.py mot \
--exp_id 'mot16_ft_jla_10_60' \
--arch 'rnnforecast_34' \
--load_model '../models/jla.pth' \
--data_cfg '../src/lib/cfg/mot16.json' \
--batch_size '8' \
--num_epochs '20' \
--lr_step '15' \
--num_workers '8' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--forecast \
--use_embedding \

python track.py mot --load_model ../exp/mot/mot16_ft_jla_10_60/model_20.pth --conf_thres 0.4 --val_mot16 --exp_id mot16_ft_jla_10_60 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60
