1. audioset & spcv2

    For audioset and spcv2, we use LMDBDataset, so you need to convert datasets to lmdb-format first. We provide  scripts for converting. 


    After downloading spcv2 dataset, follow the steps below:    
    ```
    # scripts are in audiossl/scripts/dataset_preprocess
    python speech_command_v2.py  {path_to_spcv2 }
    # after processing, lmdb files are stored in {path_to_spcv2}/lmdb
    ```
    
    After downloading audioset dataset, follow the steps below:    

    Put downloaded files under the following structure
    ```
    - audioset
        - audio
            - balanced_train_segments
                - *.wav
                - ...
            - unbalanced_train_segments
                - *.wav
                - ...
            - eval_segments
                - *.wav
                - ...
        - csv
            - banlanced_train_segments.csv
            - unbanlanced_train_segments.csv
            - eval_segments.csv
    ```

    Then

    ```
    python audioset.py  {path_to_audioset }
    # Since audioset is large, you need to wait a long time ...
    # after processing, lmdb files of unbanlanced set and balanced set are stored in {path_to_audioset}/lmdb_ub and {path_to_audioset}/lmdb_b respectively
    ```




1. Voxceleb1

    see "SID: Speaker Identitification" part in https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md 

1. nsynth & us8k

   No preprocessing, just download and unzip dataset


    