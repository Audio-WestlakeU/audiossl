source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2  voxceleb1 ~/dataset/voxceleb1 $3  $4
}

for lr in 1e-1 5e-2 2e-2 
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr}
done
