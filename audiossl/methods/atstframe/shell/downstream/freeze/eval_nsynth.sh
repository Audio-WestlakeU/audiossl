source ./eval_func.sh

eval_nsynth()
{
  echo $1 $2 $3
  eval $1 $2  nsynth ~/dataset/nsynth $3  $4 $5
}

for lr in 5e-4 2.5e-4 1e-3
do
eval_nsynth  ${n_last_blocks} ${batch_size}  $1  $lr student
eval_nsynth  ${n_last_blocks} ${batch_size}  $1  $lr teacher
done

