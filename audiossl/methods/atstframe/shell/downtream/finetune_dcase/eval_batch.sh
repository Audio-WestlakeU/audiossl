
source eval_env.sh

DEBUG=0
export DEVICE=3,4
bash eval_spcv2.sh $1  
bash eval_audioset_b.sh $1 
export DEVICE=3,7
bash eval_voxceleb1.sh $1 
wait
