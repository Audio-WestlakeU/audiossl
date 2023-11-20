source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2  audioset_b /faster/lixian/audioset_b $3  $4 $5 $6 $7 $8 $9 ${10} 
}

for lr in  3.0 
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 200 5 True 0.5 /home/lixian/audiossl/ckpts/base/audioset_b/last_blocks_1_batchsize256_lr1.0_mixup_True_max_epochs200_rrcTrue_layerwiselrTrue_maskaugFalse/last.ckpt $2
done
