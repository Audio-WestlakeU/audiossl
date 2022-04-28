# Audio Self Supervised Learning 

## install

```
    git clone https://github.com/Audio-WestlakeU/audiossl
    cd audiossl
    pip install .
```


## methods

    1. ATST: Audio Representation Learning with Teacher-Student Transformer

        goto audiossl/methods/atst

## Datasets

One of the difficult parts of doing research on audio self-supervised learning is that you need to evaluate pretrained model on diverse downstream datasets. Audiossl implements an unified dataset interface, which stores meta data and initialization entry of datasets in registry. 


1. list available datasets

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

2.  use a dataset

* data preparation 

   see audiossl/methods/atst/docs/data_prep.md

* get a dataset

```python
from audiossl import datasets
dsinfo=dataset.get_dataset("nsynth")
ds = dsinfo.creat_fn(PATH_DATASET,split="train",transform=None,target_transform=None)
```
3. transformations

    see audiossl.transforms

