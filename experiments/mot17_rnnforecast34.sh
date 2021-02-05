cd src

# python train.py mot \
# --exp_id 'mot17_rnnforecast34' \
# --arch 'rnnforecast_34' \
# --load_model '../models/ctdet_coco_dla_2x.pth' \
# --data_cfg '../src/lib/cfg/mot17.json' \
# --batch_size '8' \
# --gpus '0,1' \
# --num_epochs '30' \
# --past_length '10' \
# --future_length '60' \
# --forecast \
# --val_mot20 


# python track.py mot \
# --exp_id 'mot17_rnnforecast34' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mot17_rnnforecast34/model_last.pth' \
# --conf_thres '0.6' \
# --batch_size '8' \
# --gpu '0,1' \
# --forecast \
# --val_mot17 

python train.py mot \
--exp_id 'mot17_rf34_emb_p_10_f_60_aug' \
--arch 'rnnforecast_34' \
--load_model '../models/ctdet_coco_dla_2x.pth' \
--data_cfg '../src/lib/cfg/mot17.json' \
--batch_size '8' \
--gpus '0,1' \
--num_epochs '30' \
--past_length '10' \
--future_length '60' \
--forecast \
--use_embedding 

python track.py mot \
--exp_id 'mot17_rf34_emb_p_10_f_60_aug' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot17_rf34_emb_p_10_f_60_aug/model_last.pth' \
--conf_thres '0.4' \
--forecast \
--val_mot17 \
--use_embedding

python track.py mot \
--exp_id 'mot17_rf34_emb_p_10_f_60_aug' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot17_rf34_emb_p_10_f_60_aug/model_last.pth' \
--conf_thres '0.4' \
--forecast \
--val_mot20 \
--use_embedding

python track.py mot \
--exp_id 'test_mot17_rf34_emb_p_10_f_60_aug' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot17_rf34_emb_p_10_f_60_aug/model_last.pth' \
--conf_thres '0.4' \
--forecast \
--test_mot17 \
--use_embedding

cd ..