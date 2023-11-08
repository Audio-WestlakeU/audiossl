cd ../../../downstream
gpu_id='0,1,2,3'
arch="beats"
lr_scale=1
max_epochs=100
bsz=64
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}
    python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --pretrained_ckpt_path ./comparison_models/ckpts/BEATs_iter3.pt \
    --dcase_conf "./utils_as_strong/conf/patch_160.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./logs/as_strong_407/" \
    --prefix "_lr_${lr}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} # \
    # --freeze_mode
done
