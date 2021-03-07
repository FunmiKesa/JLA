cd src

# CUDA_VISIBLE_DEVICES='0,1' python train.py mot \
# --exp_id 'mot17_ft_crowd_jla_10_30' \
# --arch 'rnnforecast_34' \
# --load_model '../models/crowdhuman_dla34.pth' \
# --data_cfg '../src/lib/cfg/mot17.json' \
# --forecast \
# --use_embedding \
# --batch_size '16' \
# --num_epochs '30' \
# --num_workers '8' \
# --gpus '0,1' \
# --past_length '10' \
# --future_length '30' \
# --save_all \
# --multiprocessing_distributed \
# --dist-url tcp://127.0.0.1:55062 \
# --dist-backend 'nccl' \
# --world-size '1' \
# --rank '0' \

python track.py mot --load_model ../exp/mot/mot17_ft_crowd_jla_10_30/model_30.pth --conf_thres 0.4 --val_mot17 --exp_id mot17_ft_crowd_jla_10_30 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 30

cd ..
