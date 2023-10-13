source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  voxceleb1 ~/dataset/voxceleb1 $3  $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
}

for lr in 1e-1  5e-2
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True 0.75  False 0.5 10 False teacher
eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True 0.75  False 0.5 10 False student
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  ${lr} 50 5 False False True False
done
