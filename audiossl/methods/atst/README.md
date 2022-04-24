# Audio Representation Learning with Teacher-Student Transformer


## data preparation

    go to docs/data_prep.md
    
    

## pretrain


    1. train a small model
        1. bash train_small.sh data_path ckpt_path

    2. train a base model
        1. bash train_base.sh data_path ckpt_path
    
    You may need to modify "nproc" and "device" in train_*.sh according to your own environment.
    

## downstream evaluation

1. linear evaluation

    1. go to shell/downstream/freeze

    2. modify data path in eval_{task}.sh 

    3. modify enviroment in eval_env.sh

    4. run eval_batch.sh



2. finetuning

    1. go to shell/downstream/finetune

    2. modify data path in eval_{task}.sh 

    3. modify enviroment in eval_env.sh

    4. run eval_batch.sh
