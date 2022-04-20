cd src
EXP_ID=${1:-mot20_ft_jla_10_60}
ARCH=${2:-rnnforecast_34}
DESC=${3:-training on mot20}

python train.py mot \
--exp_id "$EXP_ID" \
--arch "$ARCH" \
--desc "$DESC" \
--load_model '../models/jla.pth' \
--data_cfg '../src/lib/cfg/mot20.json' \
--batch_size '8' \
--num_epochs '20' \
--lr_step '15' \
--num_workers '8' \
--gpus '0,1' \
--past_length '10' \
--future_length '60' \
--forecast \
--use_embedding \
--save_all \


python track.py mot --load_model "../exp/mot/$EXP_ID/model_20.pth" --conf_thres 0.3 --val_mot20 --exp_id "$EXP_ID" --arch "$ARCH"  --past_length 10 --future_length 2 --forecast --no_kf --use_embedding >> "$EXP_ID.txt"

cd ..
