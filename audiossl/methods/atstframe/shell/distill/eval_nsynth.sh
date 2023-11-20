source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  nsynth ~/dataset/nsynth $3  $4 $5 $6 $7 $8 $9 ${10}
}

for lr in   1e-2 2e-2
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 True 0.5  /home/lixian/models/from1.2/clip_base/nsynth/last_blocks_1_batchsize128_lr1e-3_mixup_True_max_epochs50_rrcTrue_layerwiselrTrue_maskaugFalse_alpha0.5/last.ckpt $2 &
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True False
done
wait
