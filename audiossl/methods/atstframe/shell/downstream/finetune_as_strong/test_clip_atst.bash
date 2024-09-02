cd ../../../downstream
gpu_id='0,'
arch="clipatst"
lr_scale=1.0
bsz=64
max_epochs=40
lr="1e-1"
test_ckpt="YOUR PATH HERE"
echo test: ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./comparison_models/ckpts/clip_atst.ckpt" \
    --dcase_conf "./utils_as_strong/conf/frame_40.yaml" \
    --dataset_name "as_strong" \
    --save_path "./logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_clip_atst" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
