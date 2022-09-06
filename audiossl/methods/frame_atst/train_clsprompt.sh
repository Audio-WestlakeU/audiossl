warmup_steps=1960
max_steps=58600
batch_size=256
subset=200000
ema=0.99
lr=5e-4
data_path=$1
framemodel=$2
org_save_path=$3
save_path=$3/ema${ema}steps${max_steps}
nproc=4
devices=0,1,2,3
nprompt=1

mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$devices python train_cls.py \
	            --framemodel $framemodel \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
		    --nprompt $nprompt \
                    --learning_rate $lr > $save_path/log.txt

warmup_steps=1960
max_steps=58600
batch_size=256
subset=200000
ema=0.999
lr=5e-4
data_path=$1
framemodel=$2
save_path=$3
nproc=4
devices=0,1,2,3
nprompt=1

save_path=$3/ema${ema}steps${max_steps}

mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$devices python train_cls.py \
	            --framemodel $framemodel \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
		    --nprompt $nprompt \
                    --learning_rate $lr > $save_path/log.txt
