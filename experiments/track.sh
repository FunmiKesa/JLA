cd src
# python track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/rnnforecast_34_16/model_last.pth --conf_thres 0.6 --val_mot15 --batch_size 12 --gpu 2,3 --exp_id rnnforecast_34_16

# python train.py mot --load_model /home/funmi/Experiments/FairMOT/models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 12 --gpus 0,1 --forecast --num_epochs 30 --arch rnnforecast_34 --exp_id rnnforecast_34_mask

# python train.py mot --load_model /home/funmi/Experiments/FairMOT/models/ctdet_coco_dla_2x.pth --data_cfg ../src/lib/cfg/mot16.json --batch_size 12 --gpus 0,1 --forecast --num_epochs 30 --arch rnnforecast_34 --exp_id reverse_past_bboxes

# CUDA_VISIBLE_DEVICES='2' python /home/funmi/Experiments/FairMOT/src/track.py mot --load_model /home/funmi/Experiments/FairMOT/exp/mot/mot17_rf34_emb_p_10_f_60/model_last.pth --conf_thres 0.4 --val_mot17 --batch_size 8 --forecast --exp_id mot17_rf34_emb_p_10_f_60 --arch rnnforecast_34

CUDA_VISIBLE_DEVICES='2' python track.py mot \
--exp_id 'mot17_rf34_emb_p_10_f_60_c0.4' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot17_rf34_emb_p_10_f_60/model_last.pth' \
--conf_thres '0.4' \
--past_length '10' \
--future_length '60' \
--forecast \
--val_mot20 \
--use_embedding

cd ..