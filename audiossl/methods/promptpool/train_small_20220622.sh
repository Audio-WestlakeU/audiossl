warmup_steps=1300
max_steps=39100
batch_size=128
subset=200000
ema=0.998
lr=3e-4
nproc=4
data_path=$1
arg_save_path=$2
devices=0,1,2,3


#################################################################################################
###################
warmup_steps=489
max_steps=14700
batch_size=1024
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=False
aug_tea=False
aug_stu=True
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --use_cls $use_cls \
		    --unmask_for_cls $unmask_for_cls \
		    --crop_ratio $crop_ratio \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt

#################################################################################################
###################
warmup_steps=980
max_steps=29400
batch_size=512
use_cls=2
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=True
aug_stu=True
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --use_cls $use_cls \
		    --unmask_for_cls $unmask_for_cls \
		    --crop_ratio $crop_ratio \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt


#################################################################################################
###################
warmup_steps=980
max_steps=29400
batch_size=512
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=False
aug_stu=True
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --use_cls $use_cls \
		    --unmask_for_cls $unmask_for_cls \
		    --crop_ratio $crop_ratio \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt




#################################################################################################
###################
warmup_steps=980
max_steps=29400
batch_size=512
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=True
aug_stu=True
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --use_cls $use_cls \
		    --unmask_for_cls $unmask_for_cls \
		    --crop_ratio $crop_ratio \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt

#################################################################################################
###################
warmup_steps=489
max_steps=14700
batch_size=1024
use_cls=2
unmask_for_cls=True
crop_ratio=0.6
symmetric=False
aug_tea=True
aug_stu=True
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --use_cls $use_cls \
		    --unmask_for_cls $unmask_for_cls \
		    --crop_ratio $crop_ratio \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt

#################################################################################################
###################
warmup_steps=980
max_steps=29400
batch_size=512
use_cls=3
unmask_for_cls=True
crop_ratio=0.6
max_steps=1000000
symmetric=True
aug_tea=True
aug_stu=True
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}_1000000
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$devices python train.py \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --use_cls $use_cls \
		    --unmask_for_cls $unmask_for_cls \
		    --crop_ratio $crop_ratio \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt
