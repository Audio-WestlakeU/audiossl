from torchvision import transforms 
import torchaudio
import torch
import numpy as np
import time
from torch.nn import functional as F


class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'
class Identity(CustomAudioTransform):
    def __call__(self,signal):
        return signal
    
class GaussianNoise(CustomAudioTransform):
    def __init__(self,g):
        self.g = g
    def __call__(self,signal):
        return signal + self.g * torch.randn_like(signal)

class PadToSize(CustomAudioTransform):
    def __init__(self, size:int):
        self.size = size

    def __call__(self, signal):

        if signal.shape[1] < self.size :
            signal = F.pad(signal, (0, self.size-signal.shape[1]))
        return signal

class ToSizeN(CustomAudioTransform):
    def __init__(self, size:int):
        self.size = size

    def __call__(self, signal):
        n = signal.shape[1]//self.size
        m = signal.shape[1] % self.size
        if m > self.size//2 or n==0:
            signal = F.pad(signal, (0, self.size*(n+1)-signal.shape[1]))
        else:
            signal = F.pad(signal, (0, self.size*n-signal.shape[1]))
        return signal

class CentralCrop(CustomAudioTransform):
    def __init__(self, size:int, pad:bool=True):
        self.size = size
        self.pad = pad

    def __call__(self, signal):

        if signal.shape[1] < self.size :
            if self.pad:
                signal = F.pad(signal, (0, self.size-signal.shape[1]))
            return signal

        start = (signal.shape[1] - self.size) // 2
        return signal[:, start: start + self.size]

class RandomCrop(CustomAudioTransform):
    def __init__(self, size:int, pad:bool = True):
        self.size = size
        self.pad = pad

    def __call__(self, signal):
        if signal.shape[1] < self.size :
            if self.pad:
                signal = F.pad(signal, (0, self.size-signal.shape[1]))
            return signal
        start = np.random.randint(0, signal.shape[-1] - self.size + 1)
        return signal[:, start: start + self.size]
    

class Normalize(CustomAudioTransform):
    def __init__(self,std_mean=None,reduce_dim=None):
        self.std_mean = std_mean
        self.reduce_dim = reduce_dim
    def __call__(self,input):
        """
        assuming input has shape [batch,nmels,time]
        """
        std,mean = None,None
        if self.std_mean is None:
            if self.reduce_dim is not None:
                std,mean = torch.std_mean(input,dim=self.reduce_dim,keepdim=True)
            else:
                std,mean = torch.std_mean(input)
        else:
            std,mean = self.std_mean
        output = input - mean 
        output = output / (std + 1e-6)
        return output

class MinMax(CustomAudioTransform):
    def __init__(self,min,max):
        self.min=min
        self.max=max
    def __call__(self,input):
        min_,max_ = None,None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_)/(max_- min_) *2. - 1.
        return input

class div(CustomAudioTransform):
    def __init__(self,value=100):
        self.value = value
    def __call__(self,input):
        input /= 100
        return input
