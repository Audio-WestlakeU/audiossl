
from audiossl import datasets
from audiossl.datasets import Nsynth,Urbansound8k,SpeakerClassifiDataset,IEMOCAPDataset
from pathlib import Path
import os
import torch

def load_dataset(dataset_name,data_path,fold,transform):
    if dataset_name == "spcv2":
        dataset_train = datasets.LMDBDataset(data_path,split="train",subset=1000,transform=transform) 
        dataset_val = datasets.LMDBDataset(data_path,split="valid",subset=200000,transform=transform) 
        dataset_test = datasets.LMDBDataset(data_path,split="eval",subset=200000,transform=transform) 
    elif dataset_name == "nsynth":
        dataset_train=Nsynth(data_path,split="train",transform=transform)
        dataset_val=Nsynth(data_path,split="valid",transform=transform)
        dataset_test=Nsynth(data_path,split="test",transform=transform)
    elif dataset_name == "us8k":
        dataset_train=Urbansound8k(data_path,split="train",valid_fold=fold,transform=transform)
        dataset_val=Urbansound8k(data_path,split="test",valid_fold=fold,transform=transform)
        dataset_test=Urbansound8k(data_path,split="test",valid_fold=fold,transform=transform)

    elif dataset_name == "voxceleb1":
        dataset_train=SpeakerClassifiDataset("train",
                                                       Path(data_path),
                                                       Path(os.path.join(data_path,"iden_split.txt")),
                                                       transform=transform)
        dataset_val=SpeakerClassifiDataset("dev",
                                                       Path(data_path),
                                                       Path(os.path.join(data_path,"iden_split.txt")),
                                                       transform=transform)
        dataset_test=SpeakerClassifiDataset("test",
                                                       Path(data_path),
                                                       Path(os.path.join(data_path,"iden_split.txt")),
                                                       transform=transform)
    elif dataset_name == "iemocap":
        dataset_train=IEMOCAPDataset(Path(data_path),
                                            Path(os.path.join(data_path,"meta_data","Session{}".format(fold),"train_meta_data.json")),
                                            transform=transform)
        dataset_val=IEMOCAPDataset(Path(data_path),
                                            Path(os.path.join(data_path,"meta_data","Session{}".format(fold),"test_meta_data.json")),
                                            transform=transform)
        dataset_test=dataset_val
    elif dataset_name == "audioset_b":
        dataset_train=datasets.LMDBDataset(data_path,split="train",subset=3000000,transform=transform)
        dataset_val=datasets.LMDBDataset(data_path,split="valid",transform=transform)
        dataset_test=datasets.LMDBDataset(data_path,split="eval",transform=transform)

    elif dataset_name == "audioset":
        dataset_train=datasets.LMDBDataset(data_path,split="train",subset=3000000,transform=transform)
        dataset_val=datasets.LMDBDataset(data_path,split="valid",transform=transform)
        dataset_test=datasets.LMDBDataset(data_path,split="eval",transform=transform)
    else:
        raise NotImplementedError
    return dataset_train,dataset_val,dataset_test

def extract_embedding(model, loader, n, use_cls, avgpool, args,load_args):
    header = 'extracting embedding:'
    X = []
    Y = []
    features = None
    i=0
    max_global_len=load_args.global_len[1] if "global_len" in load_args else 6
    for ((inp, length), target) in loader:
        i+=1
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        length = length.cuda(non_blocking=True)
        patch_length = (inp.shape[2]//model.patch_h) * ((length - length%model.patch_w)//model.patch_w)

        if args.dataset_name == "spcv2":
            target = torch.argmax(target,dim=-1)

        # forward
        with torch.no_grad():
            output=[]
            if load_args.arch.startswith("ast"):
                cls , avg= model.get_intermediate_layers_chunks(inp, length, n,chunk_len=int(max_global_len*101))
                if use_cls:
                    output = cls
                if avgpool:
                    if args.last_avgpool:
                        output.extend(avg)
                    else:
                        output.append(avg[-1])


            feats = torch.cat(output, dim=-1)


        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        feats_l = list(feats_all.unbind(0))
        feats_all_reduce = torch.distributed.all_gather(feats_l, feats, async_op=True)
        feats_all_reduce.wait()

        target_all = torch.empty(
            dist.get_world_size(),
            *target.size(),
            dtype=target.dtype,
            device=target.device,
        )
        target_l = list(target_all.unbind(0))
        target_all_reduce = torch.distributed.all_gather(target_l, target, async_op=True)
        target_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            X.append(torch.cat(feats_l,dim=0).cpu())
            Y.append(torch.cat(target_l,dim=0).cpu())


        torch.cuda.synchronize()
        time.sleep(0.1)
    # gather the stats from all processes
    X=torch.cat(X,dim=0)
    Y=torch.cat(Y,dim=0)
    print(f"Storing features into tensor of shape {X.shape},{Y.shape}")
    return X,Y

def get_unit_sec(dataset_name,pos_type,max_global_len):
    unit_sec = 1
    if dataset_name == "spcv2":
        unit_sec = 1
    elif dataset_name == "nsynth":
        unit_sec = 4
    elif dataset_name == "us8k":
        unit_sec=4
    elif dataset_name == "voxceleb1":
        unit_sec=8.2
    elif dataset_name == "iemocap":
        unit_sec=4
    elif dataset_name == "audioset_b":
        unit_sec=6
    elif dataset_name == "audioset":
        unit_sec=6

    if pos_type == "cut": 
        unit_sec = min(unit_sec,max_global_len)
    
    return unit_sec

NUM_LABELS={
    "spcv2":35,
    "nsynth":11,
    "us8k":10,
    "voxceleb1":1251,
    "iemocap":4,
    "audioset_b":527,
    "audioset":527
}
