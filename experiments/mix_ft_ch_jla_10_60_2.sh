cd src

python train.py mot \
--exp_id 'mix_ft_ch_jla_10_60_2' \
--arch 'rnnforecast_34' \
--load_model '../models/crowdhuman_dla34.pth' \
--data_cfg '../src/lib/cfg/data.json' \
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
--dist-url tcp://127.0.0.1:55782 \
--dist-backend 'nccl' \
--world-size '1' \
--rank '0' \

python track.py mot --load_model ../exp/mot/mix_ft_ch_jla_10_60_2/model_30.pth --conf_thres 0.4 --val_mot17 --exp_id mix_ft_ch_jla_10_60_2 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60

cd ..
