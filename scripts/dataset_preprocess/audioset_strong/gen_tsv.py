import os
import pandas as pd
import soundfile as sf
from argparse import ArgumentParser

parser = ArgumentParser("Data Preprocess")
parser.add_argument("--root_path", type=str, required=True)
args = parser.parse_args()

root_path = args.root_path
os.chdir(root_path) 
if not os.path.exists("./train"):
    os.mkdir("./train")
if not os.path.exists("./eval"):
    os.mkdir("./eval")
mode = ["eval", "train"]
# standard dcase label format: filename	onset	offset	event_label
for m in mode:
    full_meta_df = pd.read_csv(f"./source/audioset_{m}_strong.tsv", delimiter="\t")
    file_list = os.listdir(f"../data/{m}/")
    print(f"Total {m} files:", len(file_list))
    full_file_list = full_meta_df['segment_id'].values
    full_file_mask = [file + ".wav" in file_list for file in full_file_list]
    meta_df = full_meta_df[full_file_mask]
    meta_df.columns = ["filename", "onset", "offset", "event_label"]
    meta_df["filename"] = meta_df["filename"].map(lambda x: x + ".wav")
    meta_df.to_csv(f"./{m}/{m}.tsv", index=False, sep="\t")
    print(f"Total {m} files after filtering:", len(meta_df))

# generate duration tsv for eval data
eval_meta = pd.read_csv(f"./eval/eval.tsv", delimiter="\t")
file_list = pd.unique(eval_meta["filename"].values)
durations = []
for file in file_list:
    wav, fs = sf.read("../data/eval/" + file)
    durations.append(min(10, len(wav) / fs))
duration_df = pd.DataFrame({"filename": file_list, "duration": durations})
duration_df.to_csv("./eval/eval_durations.tsv", index=False, sep="\t")

