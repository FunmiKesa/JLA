cd src

python train.py mot \
--exp_id 'mot17_half_ft_ch_jla_10_60_emb2' \
--arch 'rnnforecast_34' \
--load_model '../models/crowdhuman_dla34.pth' \
--data_cfg '../src/lib/cfg/mot17_half.json' \
--forecast \
--use_embedding \
--batch_size '32' \
--num_epochs '30' \
--num_workers '16' \
--gpus '0,1,2,3' \
--past_length '10' \
--future_length '60' \
--save_all \
--multiprocessing_distributed \
--dist-url tcp://127.0.0.1:55062 \
--dist-backend 'nccl' \
--world-size '1' \
--rank '0' \

python track_half.py mot --load_model ../exp/mot/mot17_half_ft_ch_jla_10_60_emb2/model_last.pth --conf_thres 0.4 --val_mot17 --exp_id mot17_half_ft_ch_jla_10_60_emb2 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60

cd ..
