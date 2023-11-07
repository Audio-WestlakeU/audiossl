cd ../../../downstream
gpu_id='6,'
arch="clipatst"
lr_scale=1.0
bsz=64
max_epochs=40
lr="1e-1"
test_ckpt=/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/clipatst_lr_5e-1_max_epohcs_100_lr_scale_0.75/checkpoint-epoch=00099.ckpt
echo test: ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}

python3 pretrain_feat_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./utils_dcase/comparison_models/ckpts/clip_atst.ckpt" \
    --dcase_conf "./conf/frame_atst_as_strong.yaml" \
    --dataset_name "as_strong" \
    --save_path "./dcase_logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_frame_atst" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
