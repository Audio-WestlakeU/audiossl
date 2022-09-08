warmup_steps=1300
max_steps=39100
batch_size=128
subset=200000
ema=0.997
lr=5e-4
data_path=$1
arg_save_path=$2
nproc=4
devices=0,1,2,3

#################################################################################################
###################
warmup_steps=1960
max_steps=58600
batch_size=256
use_cls=0
unmask_for_cls=False
crop_ratio=0.6
symmetric=True
aug_tea=False
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=4
mask_len=3
min_mask_len=2
name=mask_uniform_use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_mask_len${mask_len}_min_mask_len${min_mask_len}
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
		    --avg_blocks $avg_blocks \
		    --patch_h $patch_h \
		    --patch_w $patch_w \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --min_mask_len $min_mask_len \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  

#################################################################################################
###################
warmup_steps=1960
max_steps=58600
batch_size=256
use_cls=0
unmask_for_cls=False
crop_ratio=0.6
symmetric=True
aug_tea=False
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=4
mask_len=3
min_mask_len=1
name=mask_uniform_use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_mask_len${mask_len}_min_mask_len${min_mask_len}
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
		    --avg_blocks $avg_blocks \
		    --patch_h $patch_h \
		    --patch_w $patch_w \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --min_mask_len $min_mask_len \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  

#################################################################################################
###################
warmup_steps=1960
max_steps=58600
batch_size=256
use_cls=0
unmask_for_cls=False
crop_ratio=0.6
symmetric=True
aug_tea=False
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=4
mask_len=2
min_mask_len=1
name=mask_uniform_use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_mask_len${mask_len}_min_mask_len${min_mask_len}
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
		    --avg_blocks $avg_blocks \
		    --patch_h $patch_h \
		    --patch_w $patch_w \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --min_mask_len $min_mask_len \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  

#################################################################################################
###################
warmup_steps=1960
max_steps=58600
batch_size=256
use_cls=0
unmask_for_cls=False
crop_ratio=0.6
symmetric=True
aug_tea=False
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=4
mask_len=4
min_mask_len=1
name=mask_uniform_use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_mask_len${mask_len}_min_mask_len${min_mask_len}
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
		    --avg_blocks $avg_blocks \
		    --patch_h $patch_h \
		    --patch_w $patch_w \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --min_mask_len $min_mask_len \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  

