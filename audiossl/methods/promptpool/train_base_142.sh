arch=base
warmup_steps=1300
max_steps=39100
batch_size=128
subset=2000000
ema=0.9995
lr=2e-4
data_path=$1
arg_save_path=$2
nproc=4
devices=0,1,2,3

#################################################################################################
###################
arch="base"
warmup_steps=19800
max_steps=391000
batch_size=128
nproc=8
devices=0,1,2,3,4,5,6,7
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=False #
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=4
mask_len=5
use_mse=0
name=archbase_lr${lr}_ema${ema}
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
		    --use_mse $use_mse \
		    --patch_h $patch_h \
		    --patch_w $patch_w \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu \
		    --arch $arch > $save_path/log.txt  

exit

#################################################################################################
###################
warmup_steps=1980
max_steps=58800
batch_size=256
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=False #
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=2
mask_len=10
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_nproc${nproc}_batch_size${batch_size}_mask_len${mask_len}
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
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  
#################################################################################################
###################
warmup_steps=1980
max_steps=58800
batch_size=256
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=False #
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=3
mask_len=10
name=use_cls${use_cls}unmask_for_cls${unmask_for_cls}crop_ratio${crop_ratio}symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_nproc${nproc}_batch_size${batch_size}_mask_len${mask_len}
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
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  


#################################################################################################
###################
warmup_steps=1980
max_steps=588000
batch_size=256
use_cls=0
unmask_for_cls=True
crop_ratio=0.6
symmetric=True
aug_tea=False #
aug_stu=True
mask_ratio=0.65
mask_type="block"
avg_blocks=0
patch_h=64
patch_w=2
mask_len=10
name=last_12
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
		    --symmetric $symmetric \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt  





