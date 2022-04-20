source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2 $3 $4 $5 $6 audioset /faster/lixian/audioset $7 $8 $9 0 ${10}
}

for lr in     1.0 5e-1
do
eval_cmd $epoch ${n_last_blocks} ${avgpool} ${use_cls} ${last_avgpool} ${batch_size} 10088 $1 $2 ${lr}
done
