cd ../../../downstream
gpu_id='0,1,2,3'
arch="distill"
mode=$1
lr_scale=0.75
bsz=64
max_epochs=100
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 pretrain_feat_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path ${mode} \
    --dcase_conf "./conf/frame_atst_as_strong.yaml" \
    --dataset_name "as_strong" \
    --save_path "./dcase_logs/as_strong_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_lr_${lr}_${mode}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs}
done
