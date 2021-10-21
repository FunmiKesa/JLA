cd src
CUDA_VISIBLE_DEVICES='0,1' python train.py mot \
--exp_id 'testNonEncodedDLAfeatures' \
--desc 'Goal: test dla features without reversing the past input' \
--arch 'rnnforecast_34' \
--load_model '../models/ctdet_coco_dla_2x.pth' \
--data_cfg '../src/lib/cfg/mot17_half.json' \
--batch_size '8' \
--num_epochs '30' \
--num_workers '8' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--forecast \
--use_embedding \
# --multiprocessing_distributed \
# --dist-url tcp://127.0.0.1:5564 \
# --dist-backend 'nccl' \
# --world-size '1' \
# --rank '0' \


python track_half.py mot --load_model ../exp/mot/testNonEncodedDLAfeatures/model_last.pth --conf_thres 0.4 --val_mot17 --exp_id testNonEncodedDLAfeatures --arch rnnforecast_34 --forecast --use_embedding --no_kf --past_length 10 --future_length 60

