cd ../../../downstream
gpu_id='7,'
arch="patchssast"
lr_scale=1
bsz=64
max_epochs=100
test_ckpt=""
for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 train_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./comparison_models/ckpts/SSAST-Base-Patch-400.pth" \
    --dcase_conf "./conf/patch_160.yaml" \
    --dataset_name "as_strong" \
    --save_path "./logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_ssast" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
done
