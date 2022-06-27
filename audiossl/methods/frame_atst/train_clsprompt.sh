warmup_steps=1300
max_steps=39100
batch_size=128
subset=200000
ema=0.99
lr=5e-4
data_path=$1
framemodel=$2
save_path=$3
nproc=1
devices=4

mkdir -p $save_path
echo CUDA_VISIBLE_DEVICES=$devices python train_cls.py \
	            --framemodel $framemodel \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr #> $save_path/log.txt
