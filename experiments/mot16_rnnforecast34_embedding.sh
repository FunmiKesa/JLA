cd src

python train.py mot \
--exp_id 'mot16_rnnforecast34_embedding_60' \
--arch 'rnnforecast_34' \
--load_model '../models/ctdet_coco_dla_2x.pth' \
--data_cfg '../src/lib/cfg/mot16.json' \
--batch_size '8' \
--gpus '0,1' \
--num_epochs '30' \
--future_length '60' \
--forecast \
--use_embedding

python track.py mot \
--exp_id 'mot16_rnnforecast34_embedding_60' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot16_rnnforecast34_embedding_60/model_last.pth' \
--conf_thres '0.6' \
--batch_size '8' \
--gpu '0,1' \
--future_length '60' \
--val_mot16 \
--forecast \
--use_embedding

python track.py mot \
--exp_id 'mot16_rnnforecast34_embedding_60' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot16_rnnforecast34_embedding_60/model_last.pth' \
--conf_thres '0.6' \
--batch_size '8' \
--gpu '0,1' \
--future_length '60' \
--val_mot20 \
--forecast \
--use_embedding

cd ..