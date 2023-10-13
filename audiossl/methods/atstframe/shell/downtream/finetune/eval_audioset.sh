source ./eval_func.sh

eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  audioset /home/lixian/dataset/audioset $3  $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}
}

for lr in 5e-1 1.0
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 11 2 True False True  0.75 False 0.5 10 False
done
