# ATST-Frame

Official implemetation of ["Self-supervised Audio Teacher-Student Transformer
for Both Clip-level and Frame-level Tasks"](https://arxiv.org/abs/2306.04186), which is accepted by IEEE Transactions on Audio, Speech and Language Processing.

This work proposes two methods: ATST-Clip and ATST-Frame. Implementation of ATST-Frame is in this directory. For ATST-Clip, please goto [../atst](../atst).

## Pre-trained Checkpoints 

Click to download
- ATST-Clip
    - goto [../atst](../atst)
- ATST-Frame

    - [atstframe_samll](https://drive.google.com/file/d/1xZoOTuxV415icYONYbeFQzgrmJQf4a4B/view?usp=sharing)

    - [atstframe_base](https://drive.google.com/file/d/1bGJSZWlAIIJ6GL5Id5dW0PTB72DL-QDQ/view?usp=sharing)

## Embedding Extraction

```python
from audiossl.methods.atstframe.embedding import load_model,get_scene_embedding,get_timestamp_embedding

model = load_model("CHECKPONT_PATH")

audio = torch.randn(1,20000) # Input audio can be of shape [1,N] or [B,1,N]

"""
extract scene (clip-level) embedding from an audio clip
=======================================
args:
    audio: torch.tensor in the shape of [1,N] or [B,1,N] 
    model: the pretrained encoder returned by load_model 
return:
    emb: retured embedding in the shape of [1,N_BLOCKS*emb_size] or [B,N_BLOCKS*emb_size], where emb_size is 768 for base model and 384 for small model.

"""
emb_scene = get_scene_embedding(audio,model)

"""
Extract frame-level embeddings from an audio clip 
==================================================
args:
    audio: torch.tensor in the shape of [1,N] or [B,1,N] 
    model: the pretrained encoder returned by load_model 
return:
    emb: retured embedding in the shape of [1,T,N_BLOCKS*emb_size] or [B,T,N_BLOCKS,emb_size], where emb_size is 768 for base model and 384 for small model, and T is number of (40ms) frames.
    timestamps: timestamps in miliseconds
"""
emb_timestamp,t = get_timestamp_embedding(audio,model)


"""
By default, embeddings of 12 blocks are concatenated.

You can change N_BLOCKS 

from audiossl.methods.atstframe.embedding
embdding.N_BLOCKS=1

"""
```


## Train Downstream Tasks

- Data prepare

    See [../atst/docs/data_prep.md](../atst/docs/data_prep.md)

- Clip-level downstream tasks
    - Linear evaluation

        1. go to shell/downstream/freeze

        2. modify data path in eval_{task}.sh 

        3. modify enviroment in eval_env.sh 

        4. run eval_batch.sh ${checkpoint_file_path}



    - Finetuning

        1. go to shell/downstream/finetune

        2. modify data path in eval_{task}.sh 

        3. modify enviroment in eval_env.sh

        4. run eval_batch.sh ${checkpoint_file_path}
- Frame-level downstream tasks
    - DESED
        - please see sehll/downstream/finetune_dcase
    - Strongly labelled AudioSet
        - please see shell/downstream/finetune_as_strong

## Train ATST-Frame

- Data prepare

    See [../atst/docs/data_prep.md](../atst/docs/data_prep.md)

- Help documentation of train.py is useful to figure out the specific meaning of each argument

    ```bash
    python train.py --help

    ```
- Train a small model (using 4 GPUs)
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --arch small \
    --data_path YOUR_DATA_OATH  \
    --save_path  YOUR_MODEL_SAVE_PATH \
    --nproc 4  \
    --batch_size_per_gpu 256 \
    --warmup_steps 1950 \
    --max_steps 58500 \
    --ema 0.997 \
    --subset 200000 \
    --learning_rate 4e-4 \
    --patch_h 64 \
    --patch_w 4 \
    --pos_type cut \
    --mask_type block \
    --mask_ratio 0.65 \
    --mask_len 5 \
    --symmetric True \
    --n_mels 64 \
    --anchor_len 10 \
    --aug_tea False \
    --aug_stu True
    ```


- Train a base model (using 6 GPUs)
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --arch base \ 
    --data_path YOUR_DATA_PATH  \
    --save_path YOUR_MODEL_SAVE_PATH \ 
    --nproc 6 --batch_size_per_gpu 144 --warmup_steps 19900 \
    --max_steps 398000 \
    --ema 0.9996 \
    --subset 3000000 \
    --learning_rate 8e-5 \
    --patch_h 64 \
    --patch_w 4 \
    --mask_type block \
    --mask_ratio 0.65 \
    --mask_len 5 \
    --symmetric True \
    --n_mels 64 \
    --anchor_len 10 \
    --aug_tea False \
    --aug_stu True
    ```

## Train ATST-C2F

Besides ATST-Clip and ATST-Frame, this work also proposes a method to combine ATST-Clip and ATST-Frame through distilling knowleadge from fintuned ATST-Clip to ATST-Frame. First, finetune ATST-Clip on a downstream task; Second, fintune ATST-Frame on the same downstream task using a multi-task loss: ground truth loss + distilation loss.



- AS-2M
    ```
    python train_distill.py
    ```
- Other downstream tasks
    ```
    python train_distill_other.py
    ```
    Take Voxceleb1 dataset for example:
    ```bash
    python train_distill_other.py \
    --batch_size_per_gpu 128 \
    --dataset_name voxceleb1 \
    --data_path DATA_PATH_Of_voxceleb1
    --pretrained_ckpt_path_clip Fintuned_CKPT_PATH_ATST_Clip_voxceleb1 \
    --pretrained_ckpt_path_frame Pretrained_CKPT_PATH_ATST_Frame \
    --learning_rate 1e-2 \
    --max_epochs 50 \
    --warmup_epochs 5 \
    --mixup_training True \
    --alpha 0.5 \
    --nproc 4 \
    --save_path YOUR_MODEL_SAVE_PATH
    ```
Fintuned_CKPT_PATH_ATST_Clip_voxceleb1 is the checkpoint of the model ATST-Clip fintuned on voxceleb1