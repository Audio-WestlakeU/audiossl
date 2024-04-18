
source eval_env.sh

export DEVICE=4,5,6,7
export NPROC=4
export cmd="python ../../../downstream/train_finetune.py"
export DEBUG=0
export n_last_blocks=1
export batch_size=128

DEBUG=1
bash eval_spcv2.sh $1
exit
bash eval_us8k.sh $1
bash eval_nsynth.sh $1
exit
bash eval_audioset_b.sh $1
bash eval_fsd50k.sh $1
exit
bash eval_voxceleb1.sh $1
exit

bash eval_audioset.sh $1
exit
wait
