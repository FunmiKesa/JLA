cd src

CUDA_VISIBLE_DEVICES='0,1' python train.py mot \
--exp_id 'mot17_half_jla_10_60_no_emb' \
--arch 'rnnforecast_34' \
--load_model '../models/ctdet_coco_dla_2x.pth' \
--data_cfg '../src/lib/cfg/mot17_half.json' \
--forecast \
--batch_size '8' \
--num_epochs '30' \
--num_workers '4' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--save_all \
# --multiprocessing_distributed \
# --dist-url tcp://127.0.0.1:55062 \
# --dist-backend 'nccl' \
# --world-size '1' \
# --rank '0' \

python track_half.py mot --load_model ../exp/mot/mot17_half_jla_10_60_no_emb/model_30.pth --conf_thres 0.4 --val_mot17 --exp_id mot17_half_jla_10_60_no_emb --arch rnnforecast_34 --forecast --no_kf --past_length 10 --future_length 60 

cd ..
