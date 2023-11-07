cd ../../../downstream
gpu_id='4,5,6,7'
arch="clipatst"
lr_scale=0.75
bsz=64
max_epochs=100
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 pretrain_feat_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./utils_dcase/comparison_models/ckpts/clip_atst.ckpt" \
    --dcase_conf "./conf/frame_atst_as_strong.yaml" \
    --dataset_name "as_strong" \
    --save_path "./dcase_logs/as_strong_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_lr_${lr}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --freeze_mode
done
