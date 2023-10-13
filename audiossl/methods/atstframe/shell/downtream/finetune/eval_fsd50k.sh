source ./eval_func.sh


eval_cmd()
{
  echo $1 $2
  eval $1 $2  fsd50k  ~/dataset/fsd50k/manifest $3  $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
}

for lr in    5e-1  1.0
do
eval_cmd  ${n_last_blocks}  ${batch_size} $1  $lr 100 5 True False True 0.75 False 0.5 10 False  teacher
eval_cmd  ${n_last_blocks}  ${batch_size} $1  $lr 100 5 True False True 0.75 False 0.5 0.5 False  teacher
#eval_cmd  ${n_last_blocks}  ${batch_size} $1  $lr 100 5 True False True False
done
