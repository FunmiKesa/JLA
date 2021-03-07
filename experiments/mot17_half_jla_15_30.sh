cd src

CUDA_VISIBLE_DEVICES='0,1' python train.py mot \
--exp_id 'mot17_half_jla_15_30' \
--arch 'rnnforecast_34' \
--load_model '../models/ctdet_coco_dla_2x.pth' \
--data_cfg '../src/lib/cfg/mot17_half.json' \
--forecast \
--use_embedding \
--batch_size '16' \
--num_epochs '30' \
--num_workers '8' \
--gpus '0,1' \
--past_length '15' \
--future_length '30' \
--save_all \
# --multiprocessing_distributed \
# --dist-url tcp://127.0.0.1:55062 \
# --dist-backend 'nccl' \
# --world-size '1' \
# --rank '0' \

python track_half.py mot --load_model ../exp/mot/mot17_half_jla_15_30/model_30.pth --conf_thres 0.4 --val_mot17 --exp_id mot17_half_jla_15_30 --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 15 --future_length 30 

cd ..
