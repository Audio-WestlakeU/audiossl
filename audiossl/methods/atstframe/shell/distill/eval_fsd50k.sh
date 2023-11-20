source ./eval_func.sh


eval_cmd()
{
  echo $1 $2  
  eval $1 $2  fsd50k  ~/dataset/fsd50k/manifest $3  $4 $5 $6 $7 $8 $9 ${10}
}

for lr in  2.0
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  $lr 100 5 True 10  /home/lixian/audiossl/ckpts/base/fsd50k/last_blocks_1_batchsize256_lr5e-1_mixup_True_max_epochs100_rrcTrue_layerwiselrTrue_maskaugFalse/last.ckpt $2
done
