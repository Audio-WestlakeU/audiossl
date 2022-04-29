# Audio Self Supervised Learning 

## Install
------------------------

1. install pytorch ( version 1.10.1 or higher )

2. install audiossl

```
    git clone https://github.com/Audio-WestlakeU/audiossl
    cd audiossl
    pip install .
```


## Methods
------------------------------

1. ATST: Audio Representation Learning with Teacher-Student Transformer

    See [audiossl/methods/atst](audiossl/methods/atst)

## Datasets
--------------------------------

One of the difficult parts of doing research on audio self-supervised learning is that you need to evaluate pretrained model on diverse downstream datasets. Audiossl implements an unified dataset interface to make evaluation easier. It's also easy to implement a new dataset.


1. List available datasets

    ```python
    from audiossl import datasets
    print(datasets.list_all_datasets())
    """ output:
    voxceleb1:
    { 'creator': <function create_voxceleb1 at 0x7fbe285d0f80>,
    'multi_label': False,
    'num_folds': 1,
    'num_labels': 1251}
    us8k:
    { 'creator': <function create_us8k at 0x7fbe285d6170>,
    'multi_label': False,
    'num_folds': 10,
    'num_labels': 10}
    nsynth:
    { 'creator': <function create_nsynth at 0x7fbe285d60e0>,
    'multi_label': False,
    'num_folds': 1,
    'num_labels': 11}
    spcv2:
    { 'creator': <function create_spcv2 at 0x7fbe285d64d0>,
    'multi_label': False,
    'num_folds': 1,
    'num_labels': 35}
    audioset_b:
    { 'creator': <function create_spcv2 at 0x7fbe285d6560>,
    'multi_label': True,
    'num_folds': 1,
    'num_labels': 527}
    audioset:
    { 'creator': <function create_spcv2 at 0x7fbe285d65f0>,
    'multi_label': True,
    'num_folds': 1,
    'num_labels': 527}
    """
    ```

2.  Use a dataset

    * Data preparation 

        See audiossl/methods/atst/docs/data_prep.md

    * Get a dataset

        ```python
        from audiossl import datasets
        dsinfo=dataset.get_dataset("nsynth")
        ds = dsinfo.creat_fn(PATH_DATASET,split="train",transform=None,target_transform=None)
        ```
3. Transformations

    See audiossl.transforms

