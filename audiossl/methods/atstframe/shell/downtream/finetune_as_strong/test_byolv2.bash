cd ../../../downstream
gpu_id='0,'
arch="byola"
lr_scale=1
max_epochs=100
bsz=64
ckpt="YOUR PATH HERE"
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}
    python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --pretrained_ckpt_path "./comparison_models/ckpts/AudioNTT2022-BYOLA-64x96d2048.pth" \
    --dcase_conf "./utils_as_strong/conf/frame_40.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./logs/test_407/" \
    --prefix "_byolv2" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${ckpt}
done