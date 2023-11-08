cd ../../../downstream
gpu_id='6,'
arch="distill"
mode=$1
lr_scale=0.75
bsz=64
max_epochs=100
ckpt='/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/distill_lr_5e-1_clip->frame_lr_scale_0.75_finetune/checkpoint-epoch=00099.ckpt'
for lr in "5e-1"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    python3 train_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path ${mode} \
    --dcase_conf "./conf/frame_atst_as_strong.yaml" \
    --dataset_name "as_strong" \
    --save_path "./dcase_logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "test_distill" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${ckpt}
done
