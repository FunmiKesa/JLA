cd src

CUDA_VISIBLE_DEVICES='1' python track_half.py mot \
--exp_id 'mot17_half_jla34' \
--arch 'rnnforecast_34' \
--load_model '../exp/mot/mot17_half_jla34/model_last.pth' \
--conf_thres '0.4' \
--past_length '10' \
--future_length '60' \
--val_mot17 \
# --forecast \
# --use_embedding \

# CUDA_VISIBLE_DEVICES='1' python track.py mot \
# --exp_id 'mot17_half_jla34_full' \
# --arch 'rnnforecast_34' \
# --load_model '../exp/mot/mot17_half_jla34/model_last.pth' \
# --conf_thres '0.4' \
# --past_length '10' \
# --future_length '60' \
# --forecast \
# --val_mot17 \
# --use_embedding

cd ..