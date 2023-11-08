cd ../../../downstream
gpu_id='0,'
arch="mmd"
lr_scale=1
max_epochs=100
bsz=64
test_ckpt="YOUR PATH HERE"

echo ${arch}, learning rate: ${lr}
python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --pretrained_ckpt_path "./comparison_models/ckpts/m2d_vit_base-80x608p16x16-221006-mr6/mmd_ckpt.pth" \
    --dcase_conf "./utils_as_strong/conf/patch_160.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./logs/test_407/" \
    --prefix "_mmd" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}

