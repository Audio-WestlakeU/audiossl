
source eval_env.sh

export DEVICE=4,5,6,7
export NPROC=4
export batch_size=128

DEBUG=0
bash eval_audioset.sh $1
bash eval_spcv2.sh $1
bash eval_us8k.sh  $1
bash eval_nsynth.sh $1

bash eval_voxceleb1.sh $1
#bash eval_audioset_b.sh $1
#bash eval_fsd50k.sh $1
