import torch
import random
import numpy as np

from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, RandomAffine, Resize, Grayscale
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset

# 设置数据加载器Worker的随机数种子
# 参数: worker_id(worker的id)
# 返回: 无
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# 根据数据分布参数将训练数据集分配给不同的客户端
# 参数: num_client(客户端数), train_set(训练数据集), bs(批次大小), num_workers(Workers数), alpha(分布参数), shuffle(是否打乱)
# 返回: client_loaders - 每个客户端的数据加载器列表
def split_data(num_client, train_set, bs=256, num_workers=10, alpha=5, shuffle=True):
    labels = np.array(train_set.targets if hasattr(train_set, 'targets') else [s[1] for s in train_set.samples])
    num_classes = np.max(labels) + 1

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = defaultdict(list)

    for c in range(num_classes):
        idx = class_indices[c]
        proportions = np.random.dirichlet(alpha=np.ones(num_client) * alpha)
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        split_idx = np.split(idx, proportions)
        for i in range(num_client):
            client_indices[i].extend(split_idx[i])

    g = torch.Generator()
    g.manual_seed(42)
    client_loaders = []
    for i in range(num_client):
        indices = client_indices[i]
        if len(indices) == 0:
            indices = np.random.choice(len(train_set), size=10, replace=False)
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        client_loaders.append(loader)
    return client_loaders

# 为NLP任务分配数据集到不同客户端
# 参数: num_client(客户端数), dataset(NLP数据集), bs(批次大小), num_workers(Workers数), alpha(分布参数)
# 返回: (client_loaders, complete_loader) - 客户端数据加载器列表和完整数据集加载器
def split_nlp_data(num_client, dataset, bs=32, num_workers=4, alpha=5):
    labels = np.array(dataset['label'])
    num_classes = np.max(labels) + 1

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = defaultdict(list)

    for c in range(num_classes):
        idx = class_indices[c]
        proportions = np.random.dirichlet(np.ones(num_client) * alpha)
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        split_idx = np.split(idx, proportions)
        for i in range(num_client):
            client_indices[i].extend(split_idx[i])

    g = torch.Generator()
    g.manual_seed(42)
    client_loaders = []
    for i in range(num_client):
        indices = client_indices[i]
        if len(indices) == 0:
            indices = np.random.choice(len(dataset), size=10, replace=False)
        subset = dataset.select(indices).with_format("torch")
        loader = DataLoader(subset, batch_size=bs, shuffle=True, collate_fn=default_data_collator,
                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        client_loaders.append(loader)
    complete_loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=default_data_collator, 
                                 num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    return client_loaders, complete_loader

# 加载MNIST数据集并分配给客户端
# 参数: num_client(客户端数), root(数据路径), bs(批次大小), alpha(分布参数), shuffle(是否打乱)
# 返回: (train_loaders, validate_loader, complete_train_loader)
def mnist(num_client, root='/dataset/mnist', bs=256, alpha=5, shuffle=False):
    transform_train = Compose([
        RandomRotation(10), 
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    train_set = MNIST(root=root, train=True, transform=transform_train, download=True)
    validate_set = MNIST(root=root, train=False, transform=transform_test)
    train_loaders = split_data(num_client, train_set, bs, alpha=alpha, shuffle=shuffle)
    complete_train_loader = DataLoader(train_set, bs, num_workers=20, shuffle=False)
    validate_loader = DataLoader(validate_set, bs, num_workers=20, shuffle=False)
    return train_loaders, validate_loader, complete_train_loader


def fashionmnist(num_client, root='/dataset/fashionmnist', bs=256, alpha=5, shuffle=False, as_rgb=False, image_size=28):
    train_transforms = [
        RandomRotation(10),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]
    test_transforms = []

    if image_size != 28:
        resize_op = Resize((image_size, image_size))
        train_transforms.append(resize_op)
        test_transforms.append(resize_op)

    if as_rgb:
        gray_to_rgb = Grayscale(num_output_channels=3)
        train_transforms.append(gray_to_rgb)
        test_transforms.append(gray_to_rgb)
        normalize_mean = (0.2860, 0.2860, 0.2860)
        normalize_std = (0.3205, 0.3205, 0.3205)
    else:
        normalize_mean = (0.2860,)
        normalize_std = (0.3205,)

    train_transforms.extend([
        ToTensor(),
        Normalize(normalize_mean, normalize_std),
    ])
    test_transforms.extend([
        ToTensor(),
        Normalize(normalize_mean, normalize_std),
    ])

    transform_train = Compose(train_transforms)
    transform_test = Compose(test_transforms)
    train_set = FashionMNIST(root=root, train=True, transform=transform_train, download=True)
    validate_set = FashionMNIST(root=root, train=False, transform=transform_test)
    train_loaders = split_data(num_client, train_set, bs, alpha=alpha, shuffle=shuffle)
    complete_train_loader = DataLoader(train_set, bs, num_workers=20, shuffle=False)
    validate_loader = DataLoader(validate_set, bs, num_workers=20, shuffle=False)
    return train_loaders, validate_loader, complete_train_loader

# 加载CIFAR10数据集并分配给客户端
# 参数: num_client(客户端数), root(数据路径), bs(批次大小), alpha(分布参数)
# 返回: (train_loaders, validate_loader, complete_train_loader)
def cifar10(num_client, root='/dataset/cifar10', bs=256, alpha=5):
    transform_train = Compose([ 
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = CIFAR10(root=root, train=True, transform=transform_train, download=True)
    validate_set = CIFAR10(root=root, train=False, transform=transform_test)
    train_loaders = split_data(num_client, train_set, bs, alpha=alpha)
    complete_train_loader = DataLoader(train_set, bs, num_workers=20, shuffle=False)
    validate_loader = DataLoader(validate_set, bs, num_workers=20, shuffle=False)
    return train_loaders, validate_loader, complete_train_loader

# 加载CIFAR100数据集并分配给客户端
# 参数: num_client(客户端数), root(数据路径), bs(批次大小), alpha(分布参数)
# 返回: (train_loaders, validate_loader, complete_train_loader)
def cifar100(num_client, root='/dataset/cifar100', bs=256, alpha=5):
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
     ])
    train_set = CIFAR100(root=root, train=True, transform=transform_train, download=True)
    validate_set = CIFAR100(root=root, train=False, transform=transform_test)
    train_loaders = split_data(num_client, train_set, bs, alpha=alpha)
    complete_train_loader = DataLoader(train_set, bs, num_workers=20, shuffle=False)
    validate_loader = DataLoader(validate_set, bs, shuffle=False)
    return train_loaders, validate_loader, complete_train_loader