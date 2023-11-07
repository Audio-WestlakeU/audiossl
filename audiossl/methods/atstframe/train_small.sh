data_path=$1
arg_save_path=$2
nproc=4
devices=0,1,2,3



#################################################################################################
###################
warmup_steps=1950
max_steps=58500
subset=200000
ema=0.997
lr=4e-4
batch_size=256
symmetric=True
aug_tea=False #
aug_stu=True
arch=small
mask_ratio=0.65
mask_type="block"
anchor_len=10
patch_h=64
patch_w=4
n_mels=64
mask_len=5
win_length=640
pos_type="cut"
name=pos_type_${pos_type}_arch_${arch}_subset_${subset}_n_mels${n_mels}_anchor_len${anchor_len}_lr_${lr}_ema_${ema}_symmetric${symmetric}aug_tea${aug_tea}aug_stu${aug_stu}mask_ratio${mask_ratio}mask_type${mask_type}avg_blocks${avg_blocks}_patch_h${patch_h}patch_w${patch_w}_nproc${nproc}_batch_size${batch_size}_mask_len${mask_len}_maxsteps${max_steps}_winlength${win_length}
save_path=${arg_save_path}/$name
###################
mkdir -p $save_path

echo CUDA_VISIBLE_DEVICES=$devices python train.py \
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
		    --pos_type $pos_type \
		    --mask_type $mask_type \
		    --mask_ratio $mask_ratio \
		    --mask_len $mask_len \
		    --symmetric $symmetric \
		    --n_mels $n_mels \
            --win_length $win_length \
		    --anchor_len $anchor_len \
		    --aug_tea $aug_tea \
		    --aug_stu $aug_stu #> $save_path/log.txt


