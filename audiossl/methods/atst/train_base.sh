arch=base
warmup_steps=15860
max_steps=318000
batch_size=256
subset=3000000
ema=0.9995
lr=2e-4
nproc=6
data_path=$1
save_path=$2
devices=0,1,2,3,4,5
mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$devices python  train.py\
                    --arch ${arch}\
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr > $save_path/log.txt
