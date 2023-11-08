cd ../../../downstream
gpu_id='3,'
arch="beats"
lr_scale=1
max_epochs=60
lr=1e-1
test_ckpt=/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/beats_all_lr_5e-1_max_epohcs_100_finetune/checkpoint-epoch=00081.ckpt

echo ${arch}, learning rate: ${lr}
python3 train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch}  \
    --pretrained_ckpt_path ./utils_dcase/comparison_models/ckpts/BEATs_iter3.pt \
    --dcase_conf "./conf/beats_as_strong.yaml" \
    --dataset_name "as_strong" \
    --batch_size_per_gpu 64 \
    --save_path "./dcase_logs/test_407/" \
    --prefix "_test_beats" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}

