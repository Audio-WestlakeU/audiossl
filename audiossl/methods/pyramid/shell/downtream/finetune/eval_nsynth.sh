
source ./eval_func.sh


eval_cmd()
{
  echo $1 $2  
  eval $1 $2  nsynth  ~/dataset/nsynth $3  $4 $5 $6 $7
}

for lr in 5e-4 2.5e-4 1e-3
do
   eval_cmd  ${n_last_blocks}  ${batch_size} $1  5e-1 50 5 True
done
