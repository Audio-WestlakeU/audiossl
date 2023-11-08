from collections import OrderedDict

def get_lab_dict(path=None):
    if path is None:
        path = "/home/shaonian/Datasets/AudioSet_strong/meta/common_labels.txt"
    with open(path, "r") as f:
        labels = f.readlines()
        labels = [l.strip() for l in labels]

    # encode labels to number
    classes_labels = OrderedDict()
    for i, label in enumerate(labels):
        classes_labels[label] = i
    return classes_labels

