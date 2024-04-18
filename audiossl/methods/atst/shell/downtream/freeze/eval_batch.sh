
source eval_env.sh

DEVICE=$2
DEBUG=0
bash eval_spcv2.sh $1
bash eval_audioset_b.sh $1
#bash eval_fsd50k.sh $1
bash eval_voxceleb1.sh $1
bash eval_nsynth.sh $1
bash eval_us8k.sh $1
#bash eval_iemocap.sh $1 linear
