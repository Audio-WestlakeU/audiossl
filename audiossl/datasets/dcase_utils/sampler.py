import numpy as np
import bisect
from torch.utils.data import DistributedSampler
from random import shuffle as sf


class ConcatDatasetSampler(DistributedSampler):
    """ConcatDatasetSampler  
    This iterative sampler is implemented to be used with ConcatDataset in pytorch.
    The index it generated is the values in the cumsum of batch sizes

    params:
        data_sources: list of datasets.
        batch_sizes: list of batch sizes.
        shuffle: shuffle the dataset when one entire dataset is fetched out once.
        mode: the dataset index for which determines the steps of each epoch. 
        (E.g. the lengths of three datasets are [1000, 2000, 3000], where the batch sizes are [125, 150, 200], 
        if mode=0, the total steps in one epoch is 1000 / 125 = 8; If the mode=2, the steps is 3000 / 200 = 15.)
    funcs:
        get_bsz: please do use this function to set the batch_size in dataloader, otherwise, the step size will be wrongly estimated.
    """


    def __init__(self, data_sources, batch_sizes, shuffle=False, mode: int = 0, drop_last=True) -> None:

        if not isinstance(data_sources, (list, tuple)):
            raise ValueError(
                "samplers should be a list or tuple of Pytorch Samplers, "
                "but got samplers={}".format(batch_sizes)
            )

        if not isinstance(batch_sizes, (list, tuple)):
            raise ValueError(
                "batch_sizes should be a list or tuple of integers, "
                "but got batch_sizes={}".format(batch_sizes)
            )

        if not len(batch_sizes) == len(data_sources):
            raise ValueError(
                "batch_sizes and samplers should be have same length")

        self.data_sources = data_sources
        # Create iterable iterator for all dataset indexes
        self.data_idxes = []
        self.data_iterators = []
        for dataset in self.data_sources:
            indexes = list(range(len(dataset)))
            sf(indexes) if shuffle else None
            self.data_idxes.append(indexes)
            self.data_iterators.append(iter(indexes))
        self.idx_count = 0

        # Start point of each dataset index
        self.offsets = [0] + np.cumsum([len(x) for x in self.data_sources]).tolist()[:-1]
        self.cum_batch_sizes = np.cumsum(batch_sizes)

        # Params
        self.mode = mode
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.idx_count = 0

        for _ in range(self.get_steps()):
                # First setup the batch size
                # Reset index counter
                if self.idx_count == self.get_bsz():
                    self.idx_count = 0

                # Select dataset
                dataset_idx = bisect.bisect_right(self.cum_batch_sizes, self.idx_count)
                self.idx_count += 1

                # Get data sample index according to the dataset index
                try:
                    sample_idx = next(self.data_iterators[dataset_idx])

                except StopIteration:
                    if self.shuffle:
                        sf(self.data_idxes[dataset_idx])
                    self.data_iterators[dataset_idx] = iter(self.data_idxes[dataset_idx])
                    sample_idx = next(self.data_iterators[dataset_idx])
                    

                yield self.offsets[dataset_idx] + sample_idx


    def __len__(self):
        return self.get_steps()

    def get_bsz(self):
        return sum(self.batch_sizes)
    
    def get_steps(self):
        # Total loop count: (select_dataset_sample_amount // select_batch_size) * sum(batch_sizes)
        batches = len(self.data_sources[self.mode]) // self.batch_sizes[self.mode]
        if not self.drop_last:
            batches += 1 if len(self.data_sources[self.mode]) % self.batch_sizes[self.mode] else 0
        return self.get_bsz() * batches


if __name__ == "__main__":
    dataset_simulate = [range(x) for x in [20, 25, 27]]
    sampler = ConcatDatasetSampler(dataset_simulate, [2, 5, 6], mode=2)

    for i in sampler:
        print(i)

