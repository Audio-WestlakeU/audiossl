import dataset

if __name__ == "__main__":

    import sys
    import torch

    path,split = sys.argv[1:]
    ds = dataset.LMDBDataset(path,split)
    for item in iter(ds):
        print(item[0].shape,item[1].shape,item[1],torch.argmax(item[1]))
    