source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3 
  eval $1 $2 $3 $4 $5 $6 iemocap /mnt/v100_data/lixian/dataset/iemocap $7 $8 $9 ${10} ${11}
}

for fold in `seq 1 5`
do
eval_cmd $epoch ${n_last_blocks} ${avgpool} ${use_cls} ${last_avgpool} ${batch_size} 10088 $1 $2 $fold 2e-3
done
