cd ../../../downstream
gpu_id='2,'
arch="mmd"
lr_scale=1
max_epochs=100
bsz=64
test_ckpt=/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/mmd_freeze_lr_5e-1_max_epohcs_100/checkpoint-epoch=00099.ckpt
for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}
    python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --dcase_conf "./conf/beats_as_strong.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu ${bsz} \
    --save_path "./dcase_logs/test_407/" \
    --prefix "_test_mmd" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
done
