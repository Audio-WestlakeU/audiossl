# Audio Representation Learning with Teacher-Student Transformer


This work focuses on the problem of segment-level gen-
eral audio SSL, and proposes a new transformer-based teacher-
student SSL model, named ATST.  Experiments have been conducted using the
large-scale Audioset  dataset for pre-training. Downstream
tasks cover all the three types of audio signals, namely audio
event, speech, and music. Ablation experiments show the effec-
tiveness of each of the proposed modules. The proposed model
as a whole achieves the new state-of-the-art results on almost all
of the downstream tasks, and surpasses other methods by a large
margin on some of the downstream tasks. For example, the ac-
curacy of speaker identification is 72% versus 40.1% without
finetuning, and 94.3% versus 80.8% after finetuning.

The full paper can be found in the preprint https://arxiv.org/abs/2204.12076

## Install

To use this repo, install audiossl first:

```
    git clone https://github.com/Audio-WestlakeU/audiossl
    cd audiossl
    pip install .
```


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


## Related projects

    1. https://github.com/s3prl/s3prl
    2. https://github.com/nttcslab/byol-a
    3. https://github.com/SarthakYadav/fsd50k-pytorch
    4. https://github.com/facebookresearch/dino

    This project learns a lot from the above projects, and some code  snippets of this project are copied or modified from them.
