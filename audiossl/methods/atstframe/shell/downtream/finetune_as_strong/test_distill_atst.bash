# Still need to provide a mode!

cd ../../../downstream
gpu_id='0,'
arch="distill"
mode=$1
lr_scale=0.75
bsz=64
max_epochs=100
ckpt='YOUR PATH HERE'

echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
python3 train_as_strong.py \
    --nproc ${gpu_id} \
    --learning_rate ${lr} \
    --arch ${arch} \
    --pretrained_ckpt_path ${mode} \
    --dcase_conf "./utils_as_strong/conf/frame_40.yaml" \
    --dataset_name "as_strong" \
    --save_path "./logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_distill" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${ckpt}

