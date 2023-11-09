# 0. To process the data with this script, we assume the directory of the data:
# |ROOT_PATH
# |--data
# |----your .wav files here
# |--meta
# |----source
# |------the raw audioset tsv files here 
# |------(audioset_eval_strong.tsv, audioset_train_strong.tsv provided in our home page)
# With this directory path, give the absolute ROOT_PATH as the bash input.
DATA_ROOT=$1
meta_path="$DATA_ROOT/meta/"
cd ./audioset_strong
# 1. Transform the raw meta data (train/eval) into DCASE format + Generate duration files for evaluation
echo "Step1: Generate raw meta data .tsv file in DCASE format"
python gen_tsv.py --root_path $meta_path

# 2. Extract the common labels shared by the train/eval datasets
echo "Step2: Extract common labels shared by the train/eval datasets"
python common_label_filtrate.py --root_path $meta_path

# 3. Remove the intersected events in a single file from the train/eval datasets
# E.g. In 1.wav, there may exists two "Dog" events, where the meta file may in the following shape:
# filename | event_label | onset | offset |
#    1.wav |    Dog      | 0.02  | 1.12   |
#    1.wav |    Dog      | 0.82  | 2.02   |
# The two same events are intersected in the timestamps, so we process them into 1 single event:
#    1.wav |    Dog      | 0.02  | 2.02   |
echo "Step3: Remove conflicted intersected events in a single file"
python intersected_event_filtrate.py --root_path $meta_path

# 4. After that, your dictory should be like:
# |ROOT_PATH
# |--data
# |----your .wav files here
# |--meta
# |----source
# |------audioset_train_strong.tsv 
# |------audioset_eval_strong.tsv
# |----train
# |------train.tsv
# |------train_common.tsv
# |------train_rm_intersect.tsv
# |----eval
# |------eval.tsv
# |------eval_common.tsv
# |------eval_durations.tsv
# |------eval_rm_intersect.tsv
# |----common_labels.txt
# 
# When fine-tuning/linear evaluation, please change the config file data path to:
# in audiossl/methods/atstframe/downstream/utils_as_strong/conf/xxx.yaml 
# data: 
#   strong_folder: "YOUR_PATH/data/train/"
#   strong_folder_44k: "YOUR_PATH/data/train/"
#   strong_train_tsv:  "YOUR_PATH/meta/train/train_rm_intersect.tsv"
#   strong_val_tsv: "YOUR_PATH/meta/eval/eval_rm_intersect.tsv"
#   test_folder: "YOUR_PATH/data/eval/"
#   test_folder_44k: "YOUR_PATH/data/eval/"
#   test_tsv: "YOUR_PATH/meta/eval/eval_rm_intersect.tsv"
#   test_dur: "YOUR_PATH/meta/eval/eval_durations.tsv"
#   label_dict: "YOUR_PATH/meta/common_labels.txt"