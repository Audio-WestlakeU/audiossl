cd ../../../downstream
gpu_id='7,'
arch="patchssast"
lr_scale=1
bsz=64
max_epochs=100
test_ckpt=/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/patchssast_freeze__double_check_lr_5e-1_max_epohcs_100/checkpoint-epoch=00099.ckpt
for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 train_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./utils_dcase/comparison_models/ckpts/SSAST-Base-Patch-400.pth" \
    --dcase_conf "./conf/beats_as_strong.yaml" \
    --dataset_name "as_strong" \
    --save_path "./dcase_logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_freeze_lr_${lr}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
done
