source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2  audioset_b /faster/lixian/audioset_b $3  $4 $5 $6 $7
}

for lr in  1.0 2.0 8e-1 5e-1 2e-1 
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} 200 5 True
done
