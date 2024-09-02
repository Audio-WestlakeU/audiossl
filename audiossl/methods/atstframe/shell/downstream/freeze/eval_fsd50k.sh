source ./eval_func.sh


eval_cmd()
{
  echo $1 $2
  eval $1 $2  fsd50k  ~/dataset/fsd50k/manifest $3  $4 $5
}

for lr in 1e-1 1.0 2.0 4.0
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  $lr teacher
eval_cmd  ${n_last_blocks}  ${batch_size} $1  $lr student
done
