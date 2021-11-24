cd src

python train.py mot \
--exp_id 'mot17_half_ft_crowd_jla_10_60' \
--arch 'rnnforecast_34' \
--load_model '../models/crowdhuman_dla34.pth' \
--data_cfg '../src/lib/cfg/mot17_half.json' \
--forecast \
--use_embedding \
--batch_size '16' \
--num_epochs '30' \
--num_workers '8' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--save_all \

python track_half.py mot --load_model ../exp/mot/mot17_half_ft_crowd_jla_10_60/model_30.pth --conf_thres 0.4 --val_mot17 --exp_id mot17_half_ft_ch_jla_10_60 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60

cd ..
