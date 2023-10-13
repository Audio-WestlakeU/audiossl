
source eval_env.sh

DEBUG=0
export DEVICE=0,1,2,3
export NPROC=4
export batch_size=256
bash eval_audioset.sh $1
#bash eval_spcv2.sh $1
bash eval_voxceleb1.sh $1
#bash eval_audioset_b.sh $1
#bash eval_fsd50k.sh $1
exit
exit
exit
wait
