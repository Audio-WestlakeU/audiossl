# Audio Representation Learning with Teacher-Student Transformer

## Introduction
---------------------

This work focuses on the problem of segment-level gen-
eral audio SSL, and proposes a new transformer-based teacher-
student SSL model, named ATST.  Experiments have been conducted using the
large-scale Audioset  dataset for pre-training. Downstream
tasks cover all the three types of audio signals, namely audio
event, speech, and music. Ablation experiments show the effec-
tiveness of each of the proposed modules. The proposed model
as a whole **achieves the new state-of-the-art results on almost all
of the downstream tasks**, and surpasses other methods by a large
margin on some of the downstream tasks. For example, the ac-
curacy of speaker identification is 72% versus 40.1% without
finetuning, and 94.3% versus 80.8% after finetuning.

![a](images/interspeech2022(a).png)
![b](images/interspeech2022(b).png)



The paper has been accepted by INTERSPEECH2022, and can be found in arxiv https://arxiv.org/abs/2204.12076

## Install
-------------------------

To use this repo, install [audiossl](../../../README.md) first:



## Data preparation
------------------------------

See [docs/data_prep.md](docs/data_prep.md)
    
    

## Downstream evaluation
---------------------------------

1. Download pretrained checkpoints

    [small.ckpt](https://checkpointstorage.oss-cn-beijing.aliyuncs.com/atst/small.ckpt)

    [base.ckpt](https://checkpointstorage.oss-cn-beijing.aliyuncs.com/atst/base.ckpt)

1. Linear evaluation

    1. go to shell/downstream/freeze

    2. modify data path in eval_{task}.sh 

    3. modify enviroment in eval_env.sh 

    4. run eval_batch.sh ${checkpoint_file_path}



1. Finetuning

    1. go to shell/downstream/finetune

    2. modify data path in eval_{task}.sh 

    3. modify enviroment in eval_env.sh (Note batch_size_per_gpu * nproc should equal to 512 to replicate the results in our paper)

    4. run eval_batch.sh ${checkpoint_file_path}

## Train your own atst model
-------------------------------------------------


    1. train a small model
        1. bash train_small.sh data_path ckpt_path

    2. train a base model
        1. bash train_base.sh data_path ckpt_path
    
You may need to modify "nproc" , "device" and also "batch_size"  in train_*.sh  to suit your needs.

## Related projects
-----------------------------------------------

    1. https://github.com/s3prl/s3prl
    2. https://github.com/nttcslab/byol-a
    3. https://github.com/SarthakYadav/fsd50k-pytorch
    4. https://github.com/facebookresearch/dino

This project learns a lot from the above projects, and some code  snippets of this project are copied or modified from them.
