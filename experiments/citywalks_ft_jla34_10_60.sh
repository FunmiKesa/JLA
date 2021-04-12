cd src
CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py mot \
--exp_id 'citywalks_ft_jla34_10_60' \
--arch 'rnnforecast_34' \
--load_model '../models/jla.pth' \
--data_cfg '../src/lib/cfg/citywalks.json' \
--batch_size '32' \
--num_epochs '20' \
--lr_step '15' \
--num_workers '16' \
--gpus '0,1,2,3' \
--past_length '10' \
--future_length '60' \
--forecast \
--use_embedding \
--save_all \
--multiprocessing_distributed \
--dist-url tcp://127.0.0.1:5564 \
--dist-backend 'nccl' \
--world-size '1' \
--rank '0' \


python track_video.py mot --load_model ../exp/mot/citywalks_ft_jla34_10_60/model_20.pth --conf_thres 0.4 --data_dir ../data/CityWalks/clips --output-format images --exp_id citywalks_ft_jla34_10_60 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 30

