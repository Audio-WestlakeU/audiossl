warmup_steps=1300
max_steps=39100
batch_size=384
subset=200000
ema=0.99
lr=5e-4
nproc=4
data_path=$1
org_save_path=$2
devices=0,1,2,3

train_len=9.0
save_path=$org_save_path/train_len${train_len}
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --train_len $train_len \
                    --subset $subset \
                    --learning_rate $lr > $save_path/log.txt

