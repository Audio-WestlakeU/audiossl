cd ../../../downstream
gpu_id='0,'
arch="audioMAE"
lr_scale=1
max_epochs=100
bsz=64
test_ckpt="YOUR OWN PATH HERE"

echo ${arch}, learning rate: ${lr}
python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --pretrained_ckpt_path "./comparison_models/ckpts/audioMAE_pretrained.pth" \
    --dcase_conf "./utils_as_strong/conf/patch_160.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./logs/test_407/" \
    --prefix "_audioMAE" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}

