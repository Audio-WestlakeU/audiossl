import torch
import numpy as np
class MixupSpecLabel:
    def __init__(self,alpha=0.5,n_memory=5000,num_classes=100):
        self.alpha = alpha
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

        l = np.random.beta(self.alpha,self.alpha,1)

        if self.memory_bank:
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