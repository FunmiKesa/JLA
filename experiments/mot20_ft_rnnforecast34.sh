cd src

CUDA_VISIBLE_DEVICES='1,2' python train.py mot \
--exp_id 'mot20_ft_rnnforecast34' \
--load_model '../exp/mot/mot17_rnnforecast34_embedding/model_last.pth' \
--num_epochs '20' \
--lr_step '15' \
--data_cfg '../src/lib/cfg/mot20.json' \
--batch_size 8 \
--arch 'rnnforecast_34' \
--forecast \
--use_embedding

python track.py mot \
--exp_id 'mot20_ft_rnnforecast34' \
--load_model '../exp/mot/mot20_ft_rnnforecast34/model_last.pth' \
--conf_thres 0.6 \
--val_mot20 \
--arch 'rnnforecast_34' \
--forecast \
--use_embedding

cd ..