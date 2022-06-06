warmup_steps=1300
max_steps=39100
batch_size=256
subset=200000
lr=5e-4
nproc=6
data_path=$1
save_path=$2
devices=2,3,4,5,6,7
mkdir -p $save_path
echo CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --subset $subset \
                    --learning_rate $lr 
CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --subset $subset \
                    --learning_rate $lr > $save_path/log.txt
