source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  audioset_b /home/lixian/dataset/audioset_b $3  $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
}

for lr in 1.0
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 200 5 True False True 0.75 False 0.5 0.5 False teacher
#eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 200 5 True False True 0.75 False 0.5 10 False
#eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 200 5 True False True False
done
