source ./eval_func.sh

eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2  audioset /faster/lixian/audioset $3  $4 $5 $6 $7
}

for lr in    5e-1 1e-1 1.0
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 20 2 True
done
