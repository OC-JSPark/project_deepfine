from itertools import chain
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets
import torch

def subset_sampler(dataset, classes, max_len):
    target_idx = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        target_idx[int(label)].append(idx)

    indices = list(chain.from_iterable(
        [target_idx[idx][:max_len] for idx in range(len(classes))]
    ))
    return Subset(dataset, indices)

def preprocess():
    train_dataset = datasets.FashionMNIST(root='/app/datasets', download=True, train=True)
    test_dataset = datasets.FashionMNIST(root='/app/datasets', download=True, train=False)

    classes = train_dataset.classes

    subset_train_dataset = subset_sampler(
        dataset=train_dataset, classes=classes, max_len=1000
    )

    subset_test_dataset = subset_sampler(
        dataset=test_dataset, classes=test_dataset.classes, max_len=100
    )

    torch.save(subset_train_dataset, '/app/subset_train_dataset.pt')
    torch.save(subset_test_dataset, '/app/subset_test_dataset.pt')

if __name__ == "__main__":
    preprocess()
