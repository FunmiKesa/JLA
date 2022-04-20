cd src
EXP_ID=${1:-mot17_half_jla_ft_ch}
ARCH=${2:-rnnforecast_34}
DESC=${3:-experimenting on half mot17}

python train.py mot \
--exp_id "$EXP_ID" \
--arch "$ARCH" \
--desc "$DESC" \
--load_model '../models/crowdhuman_dla34.pth' \
--data_cfg '../src/lib/cfg/mot17_half.json' \
--forecast \
--batch_size '8' \
--num_epochs '30' \
--num_workers '4' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--use_embedding \
--save_all \


python track_half.py mot --load_model "../exp/mot/$EXP_ID/model_30.pth" --conf_thres 0.4 --val_mot17 --exp_id "$EXP_ID" --arch "$ARCH"  --past_length 10 --future_length 2 --forecast --no_kf --use_embedding >> "$EXP_ID.txt"

cd ..
