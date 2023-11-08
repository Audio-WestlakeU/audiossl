cd ../../../downstream
gpu_id='2,'
arch="mmd"
lr_scale=1
max_epochs=100
bsz=256
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}
    python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --dcase_conf "./conf/beats_as_strong.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./dcase_logs/as_strong_407/" \
    --prefix "_freeze_lr_${lr}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --freeze_mode
done
