source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2  audioset_b /home/lixian/dataset/audioset_b $3  $4 $5
}

for lr in   1.0 2.0 5e-1
do
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} student
eval_cmd  ${n_last_blocks}  ${batch_size}  $1  ${lr} teacher
done
