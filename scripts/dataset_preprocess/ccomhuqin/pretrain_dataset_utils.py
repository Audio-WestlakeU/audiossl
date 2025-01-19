import os

import librosa
import pandas as pd
import soundfile as sf
import torchaudio
from pydub.utils import mediainfo
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import subprocess

def compute_segments(tr_tsv, new_tr_tsv, randomly=True):
    r"""
    generate segments of WINDOW_SIZE 10s, with the stride 10s
    Args:
        tr_tsv: original TSV
        new_tr_tsv: new TSV with each segment as one row, plus the start_second
    """
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    df_list = [] # will reset at fixed frequency
    total_count = 0
    if os.path.isfile(new_tr_tsv):
        print(f"Error: {new_tr_tsv} already exists! Delete it before writing.")
        return

    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path, label, audio_len, sr = row["files"], row["labels"], row["duration"], row["sample_rate"]
        upper = max(int(audio_len) - (STRIDE - 1), 1)

        # Two ways of generating segments:
        if randomly:
            # 1. generate randomly, from 0 to upper
            count = int(audio_len / STRIDE)
            start_sec_list = np.random.uniform(0, upper, count)
            for start_sec in start_sec_list:
                df_list.append([file_path, start_sec, label, sr, audio_len])
        else:
            # 2. generate sequentially, [0,10],[10,20], ...
            for curr in range(0, upper, STRIDE):
                start_sec = curr
                df_list.append([file_path, start_sec, label, sr, audio_len])
        if int(index) % 1000 == 0:
            total_count += len(df_list)
            print(f"current index is {index}, {total_count} segments in total generated. Write to file: {new_tr_tsv}")
            with open("/20A021/dataset_from_dyl/manifest_ub/segment_random/output.txt", "w") as text_file:
                text_file.write("After segmentation, dataset len: %s" % total_count)
            df_to_concat = pd.DataFrame(data=df_list, columns=["files", "start_second", "labels", "sample_rate", "duration"])
            df_list = []  # remember to clean df_list
            if os.path.isfile(new_tr_tsv):
                df_new = pd.read_csv(new_tr_tsv, delimiter="\t")
                df_new = pd.concat([df_new, df_to_concat], ignore_index=True)
                assert(total_count == df_new.shape[0])
                df_new.to_csv(new_tr_tsv, sep="\t", index=False)
            else:
                df_to_concat.to_csv(new_tr_tsv, sep="\t", index=False)

def show_specs(tr_tsv):
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    sr_dict = {}
    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path = row["files"]
        start_sec = row["start_second"]
        fig_path = f"{os.path.splitext(os.path.basename(file_path))[0]}_{start_sec}"
        if fig_path == "1041728_90900cb871447cab75c5cf295d1d9869_79.59766346820247":
            sig, fs = librosa.load(file_path, offset=start_sec, duration=10, sr=16000)
            S = librosa.feature.melspectrogram(y=sig, sr=16000, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
            to_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(to_db)

            save_fig_path = "/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/compare_" + fig_path + ".png"
            plt.savefig(save_fig_path)
            return
        # melspec_t = torchaudio.transforms.MelSpectrogram(
        #     16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        # to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        # if sr in sr_dict.keys():
        #     sr_dict[sr] += 1
        # else:
        #     sr_dict[sr] = 1
        # if int(index) % 500 == 0:
        #     print(sr_dict)
def remove_end_sec(tsv='/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/tr.tsv',
                   new_tsv='/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/tr_new.tsv'):
    df = pd.read_csv(tsv, delimiter="\t")
    df.drop(columns=['end_second']).to_csv(new_tsv, sep="\t", index=False)

def sample_df(tr_tsv, new_tr_tsv, sample_size = 2000000):
    df = pd.read_csv(tr_tsv, delimiter="\t")
    new_df = df.sample(n=sample_size).reset_index(drop=True)
    new_df.to_csv(new_tr_tsv, sep="\t", index=False)

def get_sr(tr_tsv):
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    sr_col = []
    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path = row["files"]
        sr = sf.SoundFile(file_path).samplerate
        sr_col.append(sr)
    df_tr_tsv.loc[:, "sample_rate"] = sr_col
    df_tr_tsv.to_csv(tr_tsv.replace('.tsv', '_sr.tsv'),  sep="\t", index=False)

def convert_to_wav_mono(audio_dir):
    def to_wav(filename, postfix):
        input_file = os.path.join(audio_dir, filename)
        output_file = os.path.join(audio_dir, filename.replace(postfix, ".wav"))
        if os.path.exists(output_file):
            os.remove(input_file)
            return
        # FFmpeg command to convert FLAC/mp3 to WAV and downmix to mono
        command = [
            'ffmpeg', '-i', input_file,  # Input file
            '-threads', '20',
            '-ac', '1',  # Convert to mono (1 channel)
            '-loglevel', 'quiet',
            output_file  # Output file
        ]
        # Execute the ffmpeg command
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            os.remove(input_file)
            #print(f"Successfully processed {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")

    # Loop through all FLAC files in the input directory
    for filename in tqdm(os.listdir(audio_dir)):
        if filename.endswith(".flac"):
            to_wav(filename, ".flac")
        elif filename.endswith(".mp3"):
            to_wav(filename, ".mp3")
        elif not filename.endswith(".wav"):
            print(f"{filename} is not endswith WAV") # log 写错了

def rename_to_wav(tr_tsv):
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    df_tr_tsv["files"] = df_tr_tsv["files"].str.replace('.flac', '.wav')
    df_tr_tsv["files"] = df_tr_tsv["files"].str.replace('.mp3', '.wav')
    df_tr_tsv.to_csv(tr_tsv.replace('.tsv', '_wav.tsv'), sep="\t", index=False)

def check_if_file_corrupted(tr_tsv):
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    start_seconds = df_tr_tsv["start_second"].values
    file = '/20A021/dataset_from_dyl/train-50up/audio/2070706159_600a9fbba198bf539aa9b9849fa83a16.wav'
    waveform, sr = torchaudio.load(file, normalize=True)
    print(waveform.size(1)*1.0 / sr)
    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path = row["files"]
        if file_path == file:
            print(start_seconds[index])
    #waveform, sr = torchaudio.load(file, frame_offset=13371611, num_frames=480000,normalize=True)

def gen_orig_tr_tsv(audio_dir, tr_tsv):
    df_list = []
    index, total_count = 0, 0
    # if os.path.isfile(tr_tsv):
    #     raise Exception(f"{tr_tsv} exists, delete it before generating.")
    for audio in tqdm(os.listdir(audio_dir)):
        if not audio.endswith('.wav'):
            print(f"{audio} is not a WAV file. Skip it.")
        audio_path = os.path.join(audio_dir, audio)
        duration = librosa.get_duration(path=audio_path)
        sr = librosa.get_samplerate(audio_path)
        df_list.append([audio_path, sr, duration, 'ALL'])
        index += 1
        if index % 5000 == 0:
            #total_count += len(df_list)
            df_to_concat = pd.DataFrame(data=df_list, columns=["files", "sample_rate", "duration", "labels"])
            df_list = []  # remember to clean df_list
            if os.path.isfile(tr_tsv):
                df = pd.read_csv(tr_tsv, delimiter="\t")
                df = pd.concat([df, df_to_concat], ignore_index=True)
                # if total_count != df.shape[0]:
                #     print(f"{total_count} is not equal to {df.shape[0]}")
                #     exit(1)
                df.to_csv(tr_tsv, sep="\t", index=False)
            else:
                df_to_concat.to_csv(tr_tsv, sep="\t", index=False)


if __name__ == "__main__":
    WINDOW_SIZE = 10 # anchor_len
    STRIDE = WINDOW_SIZE # No overlap
    audiodir = "/20A021/dataset_from_dyl/train-15to45/audio"
    tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/original/tr.tsv"
    random_tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/segment_random/tr.tsv"
    sample_random_tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/tr.tsv"

    #compute_segments(tr_tsv, random_tr_tsv)
    sample_df(random_tr_tsv, sample_random_tr_tsv)