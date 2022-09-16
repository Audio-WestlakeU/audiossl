import torch
from audiossl.datasets import LMDBDataset
import pyarrow as pa
import json

def find_onto(onto,id):
    for item in onto:
        if item["id"] == id:
            return item["name"]
    return "None"

if __name__ == "__main__":
    import sys

    lbl = json.load(open("lbl_map.json","r"))
    nlbl = {}
    for k,v in lbl.items():
        nlbl.update({v:k})
    onto = json.load(open("ontology.json","r"))

    query_f,data_path=sys.argv[1:]

    query_ = torch.load(query_f)
    query = query_[0]
    centroid = query_[1]

    dset = LMDBDataset(data_path,split="train",subset=200000)
    dset.keys = query.keys()

    labels = torch.zeros((centroid.shape[0],527))
    for key,value in query.items():
        sim=torch.nn.CosineSimilarity()(value,centroid)
        top = torch.topk(sim,1)
        top = top.indices[0]
        byteflow = dset.txn.get(key)
        unpacked = pa.deserialize(byteflow)
        label = torch.from_numpy(unpacked[1]).squeeze(0)
        labels[top] += label
    
    for label in labels:
        top=torch.topk(label,k=10)
        print(top.indices)
        for i in range(top.indices.shape[0]):
            id = nlbl[int(top.indices[i])]
            name = find_onto(onto,id)
            print(int(top.indices[i]),label[int(top.indices[i])],name)
    



        


    

    