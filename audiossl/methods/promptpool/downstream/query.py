
from email.mime import audio
from pytorch_lightning import LightningModule
from audiossl.modules.head import LinearHead
from audiossl.models.atst import audio_transformer
import torch
from torch import nn
from torch.nn import functional as F
from audiossl.methods.atst.downstream.utils import Metric
from itertools import chain
from audiossl.lightning.utils import EmbeddingExtractor
from audiossl.methods.atst.model import ATSTLightningModule
from pytorch_lightning import LightningDataModule
from audiossl.datasets import LMDBDataset
from torch.utils import data
from audiossl.methods.atst.downstream.transform import \
    FreezingTransform
from copy import deepcopy
def collate_fn(data):
    spec_l = []
    length_l = []
    label_l = []
    key_l = []
    for d in data:
        spec_l.append(d[0][0])
        length_l.append(d[0][1])
        label_l.append(d[1])
        key_l.append(d[2])

    max_len = max(length_l)
    for i in range(len(spec_l)):
        spec_l[i]=F.pad(spec_l[i],(0,max_len-length_l[i]))
        length_l[i]=torch.tensor(length_l[i])
        label_l[i] = torch.tensor(label_l[i])

    return (torch.stack(spec_l),torch.stack(length_l)),torch.stack(label_l),key_l


class QueryDataModule(LightningDataModule):
    def __init__(self,
                 data_path=None,
                 batch_size_per_gpu=256,
                 num_workers=10,
                 subset=200000,
                 **kwargs,
                 ):
        super().__init__()
        transform = FreezingTransform()
        self.dataset=LMDBDataset(data_path,
                                 split="train",
                                 subset=subset,
                                 transform=transform,
                                 return_key=True
                                 )
        self.batch_size=batch_size_per_gpu
        self.num_workers=num_workers
        self.save_hyperparameters()
    

    def train_dataloader(self):

        return data.DataLoader(self.dataset,
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               sampler=None,
                               collate_fn=collate_fn,
                               drop_last=True)

def get_pretraied_encoder(model_path):
    # get pretrained encoder

    s = torch.load(model_path)

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = ATSTLightningModule.load_from_checkpoint(
            model_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
    else:
        from audiossl.methods.atst.downstream.utils import \
            load_pretrained_weights
        from audiossl.models.atst.audio_transformer import AST_base, AST_small

        load_args = torch.load(model_path, map_location="cpu")["args"]
        if load_args.arch=="ast":
            pretrained_encoder = AST_small()
        else:
            pretrained_encoder = AST_base()
        load_pretrained_weights(
            pretrained_encoder, pretrained_weights=model_path, checkpoint_key="teacher")
    return pretrained_encoder

class PretrainedEncoderPLModule(LightningModule):
    def __init__(self,
                 pretrained_encoder: audio_transformer.AST,
                 chunk_len: float,
                 n_blocks: int,
                 avgpool:bool = True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.n_blocks = n_blocks
        self.avgpool = avgpool
        if avgpool:
            self.embed_dim = self.encoder.embed_dim*2*n_blocks
        else:
            self.embed_dim = self.encoder.embed_dim*n_blocks

    def forward(self, batch):
        (x, length), y, key = batch

        x = self.encoder.get_intermediate_layers_chunks(x,
                                                        length,
                                                        self.n_blocks,
                                                        self.chunk_len,
                                                        avgpool=self.avgpool)
        return  key, x


from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule,LightningModule

class EmbeddingExtractor:
    def __init__(self,
                 module:LightningModule,
                 nproc:int=1
                ):
        self.trainer = Trainer(
                            logger=False,
                            gpus=nproc,
                            #profiler="simple",
                            #max_epochs=1,
                            )
        self.module = module
    def extract(self,dataloader):
        return self.trainer.predict(self.module,dataloader)

def extract_queries(model_path,data,nproc):


    pretrained_encoder = get_pretraied_encoder(model_path)
    pretrained_module = PretrainedEncoderPLModule(pretrained_encoder,
                                                        6.,
                                                        1,
                                                        avgpool=False)
    pretrained_module.freeze()

    extracter=EmbeddingExtractor(pretrained_module,nproc=nproc)


    def get_queries(dataloader):
        result = extracter.extract(dataloader)
        
        result = [r for r in zip(*result)]

        key, x = result


        key = sum(key,[])
        x = torch.cat(x,dim=0)

        queries = {}
        for i,k in enumerate(key):
            queries.update({k:x[i]})
        return queries,x
    queries,queries_train  = get_queries(data.train_dataloader())
    queries_val,_ = get_queries(data.val_dataloader())
    queries_test,_ = get_queries(data.test_dataloader())


    queries.update(queries_val)
    queries.update(queries_test)


    """
    x = torch.cat(x,dim=0)
    y = torch.cat(y,dim=0)
    print(x.shape)
    x = torch.nn.functional.normalize(x,dim=1,p=2)
    query = torch.mean(x,dim=0)
    print(query.shape)
    """

    return queries,queries_train




