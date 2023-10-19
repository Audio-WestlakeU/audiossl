
eval()
{

        local n_last_blocks=$1
        local batch_size=$2
        local ds_name=$3
        local data_path=$4
        local pretrained_ckpt_path=$5
        local lr=$6
        local max_epochs=$7
        local warmup_epochs=$8
        local mixup_training=$9

	local mask_aug=${10}
	local rrc=${11}
	local layerwise_lr=${12}
	local alpha=${13}
        local save_path=`dirname $pretrained_ckpt_path`/${ds_name}/last_blocks_${n_last_blocks}_batchsize${batch_size}_lr${lr}_mixup_${mixup_training}_max_epochs${max_epochs}_rrc${rrc}_layerwiselr${layerwise_lr}_maskaug${mask_aug}_alpha${alpha}



        local log=$save_path/verbose.txt



        #cmd_str="CUDA_VISIBLE_DEVICES=$DEVICE $cmd --n_last_blocks $n_last_blocks
        #              --batch_size_per_gpu $batch_size
        #              --dataset_name $ds_name
        #              --data_path $data_path
        #              --pretrained_ckpt_path $pretrained_ckpt_path
        #              --fold $fold
        #              --learning_rate $lr
        #              --save_path $save_path"

        if [ $DEBUG -eq 1 ]
        then
            echo =============model===============:
            echo $pretraind_ckpt_path
            echo =============output_dir==========:
            echo $save_path
            echo ===============log===============:
            echo $log
            echo ===============command===========:
        echo CUDA_VISIBLE_DEVICES=$DEVICE $cmd --n_last_blocks $n_last_blocks \
                      --batch_size_per_gpu $batch_size\
                      --dataset_name $ds_name\
                      --data_path $data_path\
                      --pretrained_ckpt_path $pretrained_ckpt_path\
                      --learning_rate $lr\
                      --max_epochs $max_epochs\
                      --warmup_epochs $warmup_epochs\
                      --mixup_training $mixup_training\
		      --rrc $rrc\
		      --layerwise_lr $layerwise_lr\
		      --mask_aug $mask_aug\
		      --alpha $alpha\
                      --nproc $NPROC\
                      --save_path $save_path

            exit

        fi

        if [ ! -s $pretraind_ckpt_path ];then
                echo pretrained ckpt $model not exist!
                exit
        fi

        if [ ! -s $data_path ];then
                echo data $data_path not exist!
                exit
        fi
        mkdir -p $save_path


        CUDA_VISIBLE_DEVICES=$DEVICE $cmd --n_last_blocks $n_last_blocks \
                      --batch_size_per_gpu $batch_size\
                      --dataset_name $ds_name\
                      --data_path $data_path\
                      --pretrained_ckpt_path $pretrained_ckpt_path\
                      --learning_rate $lr\
                      --max_epochs $max_epochs\
                      --warmup_epochs $warmup_epochs\
                      --mixup_training $mixup_training\
		      --rrc $rrc\
		      --layerwise_lr $layerwise_lr\
		      --mask_aug $mask_aug\
		      --alpha $alpha\
                      --nproc $NPROC\
                      --save_path $save_path > $log 2>&1
}


