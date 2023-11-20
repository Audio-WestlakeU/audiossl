source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  voxceleb1 ~/dataset/voxceleb1 $3  $4 $5 $6 $7 $8 $9 ${10}
}

for lr in   5e-2
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False 10 /home/lixian/audiossl/ckpts/base/voxceleb1/last_blocks_1_batchsize256_lr5e-2_mixup_False_max_epochs50_rrcTrue_layerwiselrFalse_maskaugFalse/last.ckpt $2
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True False
done
