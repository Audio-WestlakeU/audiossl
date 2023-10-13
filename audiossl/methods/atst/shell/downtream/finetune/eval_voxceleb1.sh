source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  voxceleb1 ~/dataset/voxceleb1 $3  $4 $5 $6 $7 $8 $9 ${10} ${11}
}

for lr in  1e-1 5e-2
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True True 0.5
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True False
done
