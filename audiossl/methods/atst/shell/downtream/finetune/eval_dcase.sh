CURRENT_PATH=`pwd`
cd /mnt/home/shaonian/ATST/audiossl/audiossl/methods/atst/shell/downtream/finetune/

source ./eval_env.sh
export cmd="python /mnt/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/train_finetune_dcase.py"
export DEVICE=0
export DEBUG=0
source ./eval_func.sh

max_epochs=200
warmup_epochs=20
unfreeze_n=1
pretraind_ckpt_path='/mnt/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/dcase_logs/small.ckpt'

eval_cmd()
{
    n_last_blocks=$1
    batch_size=$2
    pretrained_ckpt_path=$3
    lr=$4
    max_epochs=$5
    warmup_epochs=$6
    unfreeze_n=$7

    echo n_last_blocks:${n_last_blocks}  batch_size:$2  lr:$4  max_epochs:$5  warmup_epochs:$6  unfreeze_n:$7

    eval ${n_last_blocks} ${batch_size}  "dcase" "/mnt/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/conf/dcase_dataset.yaml" ${pretraind_ckpt_path} ${lr} ${max_epochs} ${warmup_epochs} ${unfreeze_n}
}

for lr in 5e-1 1e-1 1e-2 1e-3
do
    eval_cmd ${n_last_blocks} ${batch_size} ${pretraind_ckpt_path} ${lr} ${max_epochs} ${warmup_epochs} ${unfreeze_n}
done

cd ${CURRENT_PATH}
