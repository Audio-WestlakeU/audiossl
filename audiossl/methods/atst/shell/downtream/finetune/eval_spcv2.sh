source ./eval_func.sh


eval_cmd()
{
  echo $1 $2
  eval $1 $2  spcv2 ~/dataset/speechcommand_v2 $3  $4 $5 $6 $7 $8 $9 ${10} ${11}
}

eval_cmd  ${n_last_blocks}  ${batch_size} $1  5e-1 50 5 True False True True 0.5
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  5e-1 50 5 True False True False
