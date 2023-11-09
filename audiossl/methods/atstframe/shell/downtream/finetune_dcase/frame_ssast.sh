cd ../../../downstream
gpu_id="6"
arch="ssast"

for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}
    python train_dcase.py \
    --nproc ${gpu_id}, \
    --learning_rate ${lr} \
    --arch ${arch} \
    --prefix _lr_${lr} \
    --pretrained_ckpt_path ./comparison_models/ckpts/SSAST-Base-Frame-400.pth \
    --dcase_conf "./utils_dcase/conf/frame_40.yaml"
done
