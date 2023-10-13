source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2  audioset_b /faster/lixian/audioset_b $3  $4
}

for lr in  100 
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr}
done
