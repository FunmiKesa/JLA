cd src

# python train.py mot \
# --exp_id 'mix_jla34_aug' \
# --arch 'rnnforecast_34' \
# --load_model '../models/ctdet_coco_dla_2x.pth' \
# --data_cfg '../src/lib/cfg/data.json' \
# --batch_size '8' \
# --gpus '0,1' \
# --forecast \
# --use_embedding

python train.py mot \
--exp_id 'mix_jla34_aug' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
--data_cfg '../src/lib/cfg/data.json' \
--batch_size '8' \
--gpus '0,1' \
--forecast \
--use_embedding \
--multiprocessing_distributed \
--dist-url tcp://127.0.0.1:5565 \
--dist-backend 'nccl' \
--world-size '1' \
--rank '0' \
--resume

# python track.py mot \
# --exp_id 'mix_jla34_aug' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.6' \
# --batch_size '8' \
# --gpu '0,1' \
# --val_mot15 \
# --forecast \
# --use_embedding

# python track.py mot \
# --exp_id 'mix_jla34_aug' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.4' \
# --batch_size '8' \
# --gpu '0,1' \
# --val_mot16 \
# --forecast \
# --use_embedding

# python track.py mot \
# --exp_id 'mix_jla34_aug' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.4' \
# --batch_size '8' \
# --gpu '0,1' \
# --val_mot17 \
# --forecast \
# --use_embedding

# python track.py mot \
# --exp_id 'mix_jla34_aug' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.4' \
# --batch_size '8' \
# --gpu '0,1' \
# --val_mot20 \
# --forecast \
# --use_embedding

# python track.py mot \
# --exp_id 'test_mix_jla34' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.4' \
# --batch_size '8' \
# --gpu '0,1' \
# --test_mot15 \
# --forecast \
# --use_embedding

# python track.py mot \
# --exp_id 'test_mix_jla34' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.4' \
# --batch_size '8' \
# --gpu '0,1' \
# --test_mot16 \
# --forecast \
# --use_embedding

# python track.py mot \
# --exp_id 'test_mix_jla34' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mix_jla34_aug/model_last.pth' \
# --conf_thres '0.4' \
# --batch_size '8' \
# --gpu '0,1' \
# --test_mot17 \
# --forecast \
# --use_embedding

cd ..