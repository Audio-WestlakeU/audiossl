cd ../../../downstream
gpu_id='6,'
arch="audioMAE"
lr_scale=1
max_epochs=100
bsz=64
test_ckpt=/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/audioMAE_lr_5e-1_max_epohcs_100_finetune/checkpoint-epoch=00056.ckpt
for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}
    python3 pretrain_feat_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --dcase_conf "./conf/beats_as_strong.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./dcase_logs/test_407/" \
    --prefix "_test_lr_${lr}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
done
