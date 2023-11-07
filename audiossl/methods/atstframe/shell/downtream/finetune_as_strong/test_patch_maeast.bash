cd ../../../downstream
gpu_id='6,'
arch="patchmaeast"
lr_scale=1.0
bsz=64
max_epochs=40
lr="1e-1"
test_ckpt=/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/patchmaeast_lr_5e-1_max_epohcs_100_finetune/checkpoint-epoch=00055.ckpt
echo test: ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}

python3 pretrain_feat_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "./utils_dcase/comparison_models/ckpts/chunk_patch_75_12LayerEncoder.pt" \
    --dcase_conf "./conf/beats_as_strong.yaml" \
    --dataset_name "as_strong" \
    --save_path "./dcase_logs/test_407/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_lr_${lr}_drop_rate_0.1_bsz_${bsz}_max_epohcs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}
