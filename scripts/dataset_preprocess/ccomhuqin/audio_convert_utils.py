import os

import librosa
import torch
import torch.utils.data as data
import torchaudio.transforms as T
from torchvision import transforms
import torchaudio
from audiossl.transforms.common import MinMax
from tqdm import tqdm
import pandas as pd
class AudioDataset(data.Dataset):
    def __init__(self, file_list, target_sample_rate=16000):
        self.file_list = file_list
        self.target_sample_rate = target_sample_rate
        self.resample = T.Resample(orig_freq=1, new_freq=target_sample_rate)
        melspec_t = T.MelSpectrogram(
            target_sample_rate, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        normalize = MinMax(min=-79.6482, max=50.6842)
        self.mel_feature = transforms.Compose(
            [melspec_t,
             to_db,
             normalize]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            self.resample.orig_freq = sample_rate
            waveform = self.resample(waveform)
        # Convert to Mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convert to Mel-Spectrogram
        mel_spec = self.mel_feature(waveform)
        return mel_spec, file_path

def save_mel_spectrogram(mel_spec, file_path, output_dir):
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.pt')
    torch.save(mel_spec, output_file)

def collate_fn(batch):
    print("")
    return {
        "mel_specs": torch.cat([x[0] for x in batch], dim=2),
        "file_paths": torch.cat([x[1] for x in batch])
    }

def batch_convert_audio_to_spec(tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/tr.tsv"):
    # Configuration
    input_dir = "/20A021/dataset_from_dyl/train-50up/audio/"
    output_dir = "/20A021/dataset_from_dyl/train-50up/melspecs/"
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 2  # Adjust based on GPU memory
    num_workers = 10  # Adjust based on CPU cores

    # Get file list
    audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                   f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.flac')]
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path, label, ext = row["files"], row["labels"], row["ext"]

    # Create dataset and dataloader
    dataset = AudioDataset(audio_files)
    dataloader = data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers,
                                 shuffle=False)

    # Processing loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count = 0
    for batch in dataloader:
        batch_mel_specs = batch["mel_specs"].to(device)
        batch_file_paths = batch["file_paths"].to(device)

        # Optionally perform further processing on GPU
        # Save the mel-spectrograms (consider saving in a loop for each file)
        for mel_spec, file_path in zip(batch_mel_specs, batch_file_paths):
            save_mel_spectrogram(mel_spec.cpu(), file_path, output_dir)
        count += len(batch_mel_specs)
        print(f"{count} converted")

    print("Processing complete.")

def convert_audio_to_spec(tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/tr.tsv",
                          target_sample_rate = 16000):
    df = pd.read_csv(tr_tsv, delimiter="\t")
    melspec_t = T.MelSpectrogram(
        target_sample_rate, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
    resampler = T.Resample(orig_freq=1, new_freq=target_sample_rate)
    for index, row in df.iterrows():
        file_path, label, ext = row["files"], row["labels"], row["ext"]
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.size(dim=1) / sample_rate
        print(file_path, duration)

        # Convert to Mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler.orig_freq = sample_rate
            resample_waveform = resampler(waveform)
            resample_mel = melspec_t(resample_waveform)
            print("Resample: ", resample_mel.size())
        orig_mel = melspec_t(waveform)
        print("Not resample: ", orig_mel.size())
        return

if __name__ == "__main__":
    convert_audio_to_spec()