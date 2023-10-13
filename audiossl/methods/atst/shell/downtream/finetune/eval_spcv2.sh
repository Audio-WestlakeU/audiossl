source ./eval_func.sh


eval_cmd()
{
  echo $1 $2
  eval $1 $2  spcv2 ~/dataset/speechcommand_v2 $3  $4 $5 $6 $7 $8 $9 ${10}
}

eval_cmd  ${n_last_blocks}  ${batch_size} $1  5e-1 50 5 True False True True
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  5e-1 50 5 True False True False
