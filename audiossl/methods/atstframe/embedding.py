import audiossl
from audiossl.methods.atstframe.model import FrameATSTLightningModule
import torch
import torchaudio
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity,CentralCrop
from torchvision import transforms

N_BLOCKS=12



melspec_t = torchaudio.transforms.MelSpectrogram(
    16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
normalize = MinMax(min=-79.6482,max=50.6842)



def load_model(model_path):
    s = torch.load(model_path)

    s = torch.load(model_path)
    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        model_path)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']

    pretrained_encoder.sample_rate = 16000
    pretrained_encoder.scene_embedding_size = pretrained_encoder.embed_dim*2*N_BLOCKS
    pretrained_encoder.timestamp_embedding_size = pretrained_encoder.embed_dim*N_BLOCKS

    pretrained_encoder.eval()
    pretrained_encoder.transform = transforms.Compose([melspec_t,
                                to_db,
                                normalize])

    return pretrained_encoder

    
def get_scene_embedding(audio,model):
    """
    extract scene (clip-level) embedding from an audio clip
    =======================================
    args:
        audio: torch.tensor in the shape of [1,N] or [B,1,N] 
        model: the pretrained encoder returned by load_model 
    return:
        emb: retured embedding in the shape of [1,N_BLOCKS*emb_size] or [B,N_BLOCKS*emb_size], where emb_size is 768 for base model and 384 for small model.

    """
    if len(audio.shape)==2: 
        audio = audio.unsqueeze(1)
    else:
        assert len(audio.shape) == 3
    
    model.to(audio.device)
    model.transform.transforms[0].to(audio.device)
    mel = model.transform(audio)
    length = torch.tensor([mel.shape[-1]]).expand(mel.shape[0])
    chunk_len=1001 # 10 secnods, consistent with the length of positional embedding
    total_len = mel.shape[-1]
    num_chunks = total_len // chunk_len + 1
    output=[]
    for i in range(num_chunks):

        start = i*chunk_len
        end = (i+1) * chunk_len
        if end > total_len:
            end = total_len
        if (end>start): #and (length +chunk_len//2  > end):
            mel_chunk=mel[:,:,:,start:end]
            len_chunk = mel_chunk.shape[-1] #if length>end+chunk_len else (length - end)
            len_chunk = torch.tensor([len_chunk]).expand(mel.shape[0]).to(audio.device)
            output_chunk = model.get_intermediate_layers(mel_chunk,len_chunk,n=12)

            output.append(output_chunk)
    output=torch.stack(output,dim=0)
    output=torch.mean(output,dim=0)


    return output


def get_timestamp_embedding(audio,model):
    """
    Extract frame-level embeddings from an audio clip 
    ==================================================
    args:
        audio: torch.tensor in the shape of [1,N] or [B,1,N] 
        model: the pretrained encoder returned by load_model 
    return:
        emb: retured embedding in the shape of [1,T,N_BLOCKS*emb_size] or [B,T,N_BLOCKS,emb_size], where emb_size is 768 for base model and 384 for small model.
        timestamps: timestamps in miliseconds
    """
    if len(audio.shape)==2: 
        audio = audio.unsqueeze(1)
    else:
        assert len(audio.shape) == 3
    

    model.to(audio.device)
    model.transform.transforms[0].to(audio.device)
    mel = model.transform(audio)
    output=[]

    chunk_len=1001 #10 secnods, consistent with the length of positional embedding

    total_len = mel.shape[-1]
    num_chunks = total_len // chunk_len + 1
    for i in range(num_chunks):

        start = i*chunk_len
        end = (i+1) * chunk_len
        if end > total_len:
            end = total_len
        if end>start:
            mel_chunk=mel[:,:,:,start:end]
            len_chunk = torch.tensor([mel_chunk.shape[-1]]).expand(mel.shape[0]).to(audio.device)

            output_chunk = model.get_intermediate_layers(mel_chunk,len_chunk,n=N_BLOCKS,scene=False)

            output.append(output_chunk)
    output=torch.cat(output,dim=1)
    length=output.shape[1]
    timestamps= (torch.arange(length)*40).float().unsqueeze(0).expand(mel.shape[0],-1)
    return output ,timestamps