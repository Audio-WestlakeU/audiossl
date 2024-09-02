# Notice that: 
# 1. You should change the ckpt path in the audiossl/methods/atstframe/downstream/comparison_models/distill_atst_module.py
# 2. You should first finetune a atst-clip/frame model to run this script.
# 3. You should give a distillation mode when you running this bash.
# 4. clip->frame = F2C (atst-clip learns from the atst-frame)
# 5. frame->clip = C2F (atst-frame learns from the atst-clip)
# 6. See finetune_distill_both_modes.bash for examples

cd ../../../downstream
gpu_id='0,1,2,3'
arch="distill"
mode=$1
lr_scale=0.75
bsz=64
max_epochs=100
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 train_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path ${mode} \
    --dcase_conf "./utils_as_strong/conf/frame_40.yaml" \
    --dataset_name "as_strong" \
    --save_path "./logs/as_strong_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_lr_${lr}_${mode}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs}
done
