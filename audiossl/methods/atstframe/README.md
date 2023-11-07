# ATST-Frame

Official implemetation of ["Self-supervised Audio Teacher-Student Transformer
for Both Clip-level and Frame-level Tasks"](https://arxiv.org/abs/2306.04186), which is currently under review at IEEE Transactions on Audio, Speech and Language Processing.

This work proposes two methods: ATST-Clip and ATST-Frame. Implementation of ATST-Frame is in this directory. For ATST-Clip, please goto [../atst](../atst).

## Pre-trained Checkpoints 

Click to download
- ATST-Clip
    - goto [../atst](../atst)
- ATST-Frame

    - [atstframe_samll](https://drive.google.com/file/d/1xZoOTuxV415icYONYbeFQzgrmJQf4a4B/view?usp=sharing)

    - [atstframe_base](https://drive.google.com/file/d/1bGJSZWlAIIJ6GL5Id5dW0PTB72DL-QDQ/view?usp=sharing)

## Train Downstream Tasks

- Data prepare

    See [../atst/docs/data_prep.md](../atst/docs/data_prep.md)

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

## Train ATST-Frame

- Data prepare

    See [../atst/docs/data_prep.md](../atst/docs/data_prep.md)

- Help documentation of train.py is useful to figure out the specific meaning of each argument

    ```
    python train.py --help

    ```
- Train a small model (using 4 GPUs)
    ```
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
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py --arch small \ 
    --data_path YOUR_DATA_PATH  \
    --save_path YOUR_MODEL_SAVE_PATH \ 
    --nproc 7 --batch_size_per_gpu 144 --warmup_steps 19900 \
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

This work proposes a method to combine ATST-Clip and ATST-Frame through distill knowleadge from fintuned ATST-Clip to ATST-Frame. First, finetune ATST-Clip on a downstream task; Second, fintune ATST-Frame on the same downstream task using a multi-task loss: ground truth loss + distilation loss.

The code nees some cleaning up.  coming soon.
- AS-2M
    ```
    python train_distill.py
    ```
- Other downstream tasks
    ```
    python train_distill_other.py
    ```
