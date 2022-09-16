
warmup_steps=1300
max_steps=39000
batch_size=384
subset=200000
ema=0.99
lr=5e-4
querymodel=~/audiossl/ckpts/base.ckpt 
data_path=$1
framemodel=$2
org_save_path=$3
nproc=4
devices=0,1,2,3
pool_size=1
prompt_len=1
select_num=1
save_path=$3/prompt${pool_size}_prompt_len${prompt_len}_select_num${select_num}

mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$devices python train_promptpool.py \
	            --framemodel $framemodel \
		    --query_model $querymodel \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
		    --pool_size $pool_size \
		    --prompt_len $prompt_len \
		    --select_num $select_num \
                    --learning_rate $lr  > $save_path/log.txt

warmup_steps=1300
max_steps=39000
batch_size=384
subset=200000
ema=0.99
lr=5e-4
querymodel=~/audiossl/ckpts/base.ckpt 
data_path=$1
framemodel=$2
org_save_path=$3
nproc=4
devices=0,1,2,3
pool_size=6
prompt_len=1
select_num=3

save_path=$3/prompt${pool_size}_prompt_len${prompt_len}_select_num${select_num}
mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$devices python train_promptpool.py \
	            --framemodel $framemodel \
		    --query_model $querymodel \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
		    --pool_size $pool_size \
		    --prompt_len $prompt_len \
		    --select_num $select_num \
                    --learning_rate $lr  > $save_path/log.txt

