source ./eval_func.sh


eval_cmd()
{
  echo $1 $2
  eval $1 $2  spcv2 ~/dataset/speechcommand_v2 $3 $4 $5 $6 $7 $8 $9 ${10}
}

eval_cmd  ${n_last_blocks}  ${batch_size} $1  2.0 50 5 True  0.5 /home/lixian/audiossl/ckpts/base/spcv2/last_blocks_1_batchsize256_lr5e-1_mixup_True_max_epochs50_rrcTrue_layerwiselrTrue_maskaugFalse/last.ckpt $2
