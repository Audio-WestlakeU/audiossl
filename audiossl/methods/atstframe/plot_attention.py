import matplotlib.pyplot as plt
import os
import torch
import torchaudio
from model import FrameATSTLightningModule
from audiossl.transforms.common import MinMax

def plot_spec(x,save_path):
    t = range(0,x.shape[0])
    f = range(0,x.shape[1])
    #plt.xlabel('frequency',fontsize=20)
    #plt.ylabel('time',fontsize=20)

    plt.axis('off')
    plt.pcolormesh(x)
    plt.xticks([])
    plt.yticks([])
    plt.margins(0,0)
    #plt.xlabel('frequency',fontsize=20)
    #plt.ylabel('time',fontsize=20)
    
    plt.savefig(save_path,dpi=500,pad_inches=-0.01,transparent=True)
    plt.close()

def plot_att(attentions,save_path,name):
    plot_spec(torch.sum(attentions[0,:,:,:],dim=0).cpu().numpy(),os.path.join(save_path,name+"att_headsum.png"))
    for i in range(attentions.shape[1]):
        plot_spec(attentions[0,i,:,:].cpu().numpy(),os.path.join(save_path,name+"att_head{}.png".format(i)))
        plt.imsave(fname=os.path.join(save_path,name+"att_head{}_imsave.png".format(i)),
                   arr=attentions[0,i].cpu().numpy(), 
                   format='png')

def wav2mel(wav_file):
    melspec_t = torchaudio.transforms.MelSpectrogram(
        16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)

    normalize = MinMax(min=-79.6482,max=50.6842)

    audio,sr = torchaudio.load(wav_file)
    assert sr == 16000
    melspec=normalize(to_db(melspec_t(audio)))
    return melspec


def get_pretraied_encoder(pretrained_ckpt_path):
    # get pretrained encoder

    s = torch.load(pretrained_ckpt_path)

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
            pretrained_ckpt_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
    return pretrained_encoder

def mel2att(mel,model):
    return model.get_last_selfattention(mel[:,:,:101].unsqueeze(0))


if __name__ == "__main__":
    import sys
    wav_file,ckpt_path=sys.argv[1:]
    save_path=os.path.dirname(ckpt_path)
    print(ckpt_path,wav_file)
    model = get_pretraied_encoder(ckpt_path)
    mel = wav2mel(wav_file)
    mel = mel[:,:,:101]
    att = mel2att(mel,model)

    os.makedirs(save_path,exist_ok=True)

    plot_spec(mel[0].cpu().numpy(),os.path.join(save_path,"mel.png"))
    print(len(att))
	
    for i,att_ in enumerate(att):
    	plot_att(att_,save_path,name="{}-".format(i))
