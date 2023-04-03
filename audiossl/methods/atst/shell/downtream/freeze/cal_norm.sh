export DEVICE=0
export cmd='python ../../../downstream/cal_norm.py'
export DEBUG=0
export n_last_blocks=12
export batch_size=1024


DEVICE=$2
DEBUG=0
bash cal_norm_audioset_b.sh $1 
