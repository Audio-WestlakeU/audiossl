
import json
import pprint
DATASET_REGISTRY={}

class DatasetInfo:
    """Placeholder for properties of dataset
    """
    def __init__(self, **kwargs): 
        self.__dict__.update(kwargs)
    def __str__(self):
        pp = pprint.PrettyPrinter(indent=2)
        return pp.pformat(self.__dict__)


def get_dataset(name:str) -> DatasetInfo:
    """
    Example::

            datasetinfo = get_dataset("spcv2")
            print(datasetinfo)
            creat_fn=datasetinfo.creator
            num_labels=datsetinfo.num_labels
            multi_label=datasetinfo.multi_label

    """
    if name in DATASET_REGISTRY.keys():
        return DATASET_REGISTRY[name]
    else:
        raise RuntimeError("dataset {} is not registered".format(name))

def list_all_datasets():
    for k,v in DATASET_REGISTRY.items():
        print("{}:\n{}".format(k,v))



def register_dataset(name,**kwargs):
    info = dict(**kwargs)

    def register_dataset_(creator):
        info.update({"creator":creator})
        if name in DATASET_REGISTRY.keys() and not (get_dataset(name).creator == creator):
            raise RuntimeError("dataset {} has been already registered".format(name))
        DATASET_REGISTRY.update({name:DatasetInfo(**info)})
        return creator
    return register_dataset_

def add_regist(name,**kwargs):
    info = dict(**kwargs)
    if name in DATASET_REGISTRY.keys() and not (get_dataset(name).creator == info["creator"]):
        raise RuntimeError("dataset {} has been already registered".format(name))
    DATASET_REGISTRY.update({name:DatasetInfo(**info)})