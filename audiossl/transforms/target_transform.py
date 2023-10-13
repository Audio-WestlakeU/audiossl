import torch
import numpy as np

def roll_mag_aug(waveform):
    waveform=waveform.numpy()
    idx=np.random.randint(len(waveform))
    rolled_waveform=np.roll(waveform,idx)
    mag = np.random.beta(10, 10) + 0.5
    return torch.Tensor(rolled_waveform*mag)

class MixupWavLabel:
    def __init__(self,mixup_ratio=0.5,n_memory=5000,num_classes=100):
        self.mixup_ratio = mixup_ratio
        self.memory_bank = []
        self.n=n_memory
        self.num_classes=num_classes

    def __call__(self,x,y):

        """convert label to one hot vector"""
        if isinstance(y,int) or (isinstance(y,torch.Tensor) and (len(y.shape)==0 or y.shape[-1]==1) ):
            if isinstance(y,int):
                y=torch.nn.functional.one_hot(torch.tensor(y),num_classes=self.num_classes)
            else:
                y=torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_classes)

        l = np.random.beta(10,10,1)

        if self.memory_bank and np.random.random()<self.mixup_ratio:
            x_,y_ = self.memory_bank[np.random.randint(len(self.memory_bank))]
            #x_mix = torch.log(x.exp()*l +x_.exp()*energy_scale(x,x_) * (1-l) + torch.finfo(x.dtype).eps)
            if x.shape[-1] == x_.shape[-1]:
                x_mix = x*l + x_*(1-l)
            elif x.shape[-1] > x_.shape[-1]:
                start = np.random.randint(0,x.shape[-1] - x_.shape[-1])
                x_mix = x.clone()

                x_mix[:,:,start:start+x_.shape[-1]] = x[:,:,start:start+x_.shape[-1]]*l + x_*(1-l)
            else:
                start = np.random.randint(0,x_.shape[-1] - x.shape[-1])

                x_mix= x*l + x_[:,:,start:start+x.shape[-1]]*(1-l)
            y_mix = y*l +y_ * (1-l)
        else:
            x_mix=x
            y_mix=y

        self.memory_bank.append((x,y))
        self.memory_bank = self.memory_bank[-self.n:]
        return x_mix.to(torch.float),y_mix.to(torch.float)

class MixupSpecLabel:
    def __init__(self,mixup_ratio=1.0,alpha=10,n_memory=5000,num_classes=100):
        self.alpha = alpha
        self.mixup_ratio = mixup_ratio
        self.memory_bank = []
        self.n=n_memory
        self.num_classes=num_classes
    def __call__(self,x,y):
        """convert label to one hot vector"""
        if isinstance(y,int) or (isinstance(y,torch.Tensor) and (len(y.shape)==0 or y.shape[-1]==1) ):
            if isinstance(y,int):
                y=torch.nn.functional.one_hot(torch.tensor(y),num_classes=self.num_classes)
            else:
                y=torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_classes)


        if self.memory_bank and np.random.random()<self.mixup_ratio:
            l = np.random.beta(self.alpha,self.alpha,1)
            x_,y_ = self.memory_bank[np.random.randint(len(self.memory_bank))]
            #x_mix = torch.log(x.exp()*l +x_.exp()*energy_scale(x,x_) * (1-l) + torch.finfo(x.dtype).eps)
            if x.shape[-1] == x_.shape[-1]:
                x_mix = x*l + x_*(1-l)
            elif x.shape[-1] > x_.shape[-1]:
                start = np.random.randint(0,x.shape[-1] - x_.shape[-1])
                x_mix = x.clone()

                x_mix[:,:,start:start+x_.shape[-1]] = x[:,:,start:start+x_.shape[-1]]*l + x_*(1-l)
            else:
                start = np.random.randint(0,x_.shape[-1] - x.shape[-1])

                x_mix= x*l + x_[:,:,start:start+x.shape[-1]]*(1-l)
            y_mix = y*l +y_ * (1-l)
        else:
            x_mix=x
            y_mix=y

        self.memory_bank.append((x,y))
        self.memory_bank = self.memory_bank[-self.n:]
        return x_mix.to(torch.float),y_mix.to(torch.float)

import torch.distributed as dist
from torch.utils.data import get_worker_info
class MixupSpecLabelAudioset:
    def __init__(self,dataset,mixup_ratio=1.0,alpha=0.5,n_memory=5000,num_classes=100):
        self.dataset = dataset
        self.mixup_ratio = mixup_ratio
        self.alpha = alpha
        self.memory_bank = []
        self.n=n_memory
        self.num_classes=num_classes
    def __call__(self,x,y):

        #rank = dist.get_rank()
        #print("rank {}".format(rank),"id {}".format(get_worker_info()),"seed {}".format(np.random.get_state()[1][0]))

        def convert_label(y):
            """convert label to one hot vector"""
            if isinstance(y,int) or (isinstance(y,torch.Tensor) and (len(y.shape)==0 or y.shape[-1]==1) ):
                if isinstance(y,int):
                    y=torch.nn.functional.one_hot(torch.tensor(y),num_classes=self.num_classes)
                else:
                    y=torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_classes)
            return y
        y = convert_label(y)

        if  np.random.random()<self.mixup_ratio:
            l = np.random.beta(self.alpha,self.alpha,1)
            index = np.random.randint(len(self.dataset))
            (x_,_),y_ = self.dataset[index]
            y_=convert_label(y_)
            if x.shape[-1] == x_.shape[-1]:
                x_mix = x*l + x_*(1-l)
            elif x.shape[-1] > x_.shape[-1]:
                start = np.random.randint(0,x.shape[-1] - x_.shape[-1])
                x_mix = x.clone()

                x_mix[:,:,start:start+x_.shape[-1]] = x[:,:,start:start+x_.shape[-1]]*l + x_*(1-l)
            else:
                start = np.random.randint(0,x_.shape[-1] - x.shape[-1])

                x_mix= x*l + x_[:,:,start:start+x.shape[-1]]*(1-l)
            y_mix = y*l +y_ * (1-l)
        else:
            x_mix=x
            y_mix=y

        return x_mix.to(torch.float),y_mix.to(torch.float)