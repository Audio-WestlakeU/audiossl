cd ../../../downstream
gpu_id='0,1,2,3'
arch="patchssast"
lr_scale=1
bsz=64
max_epochs=100
# You might need to change the timm version to finetune SSAST
# pip install --upgrade timm==0.4.5

for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 train_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./comparison_models/ckpts/SSAST-Base-Patch-400.pth" \
    --dcase_conf "./utils_as_strong/conf/patch_160.yaml" \
    --dataset_name "as_strong" \
    --save_path "./logs/as_strong_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_lr_${lr}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} # \
    # --freeze_mode
done
