data_path=$1
arg_save_path=$2
reinit_path=$3
nproc=7
devices=7,1,2,3,4,5,6




#################################################################################################
###################
warmup_steps=19900
max_steps=398000
subset=3000000
ema=0.9996
lr=8e-5
batch_size=144
symmetric=True
aug_tea=False #
aug_stu=True
arch=base
mask_ratio=0.65
mask_type="block"
anchor_len=10
patch_h=64
patch_w=4
n_mels=64
mask_len=5
name=arch_${arch}_subset_${subset}_n_mels${n_mels}_anchor_len${anchor_len}_lr_${lr}_ema_${ema}_symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_nproc${nproc}_batch_size${batch_size}_mask_len${mask_len}_maxsteps${max_steps}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

 CUDA_VISIBLE_DEVICES=$devices  python train.py \
		    --arch $arch \
                    --data_path $data_path  \
                    --save_path $save_path  \
                    --nproc $nproc \
                    --batch_size_per_gpu $batch_size \
                    --warmup_steps $warmup_steps \
                    --max_steps $max_steps \
                    --ema $ema \
                    --subset $subset \
                    --learning_rate $lr \
		    --patch_h $patch_h \
		    --patch_w $patch_w \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --symmetric $symmetric \
		    --n_mels $n_mels \
		    --anchor_len $anchor_len \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu > $save_path/log.txt

