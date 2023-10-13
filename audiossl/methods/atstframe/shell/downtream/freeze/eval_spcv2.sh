source ./eval_func.sh


eval_cmd()
{
  echo $1 $2
  eval $1 $2  spcv2 ~/dataset/speechcommand_v2 $3  $4 $5
}

eval_cmd  ${n_last_blocks}  ${batch_size} $1  2e-3 student
eval_cmd  ${n_last_blocks}  ${batch_size} $1  2e-3 teacher
