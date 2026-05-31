import os
import random
import torch
import numpy as np
from utils.data import cifar10, cifar100, mnist, fashionmnist
from transformers import get_scheduler
from model.models import vgg11, cnn
from model.alexnet import AlexNet
from model.vgg16 import VGG16
from model.resnet18 import Resnet18
from model.mnist.alexnet import AlexNet_mnist
from model.mnist.vgg16 import VGG16_mnist
from model.mnist.resnet18 import Resnet18_mnist
from copy import deepcopy

# 设置所有随机数种子确保实验可重复性
# 参数: seed(随机数种子值)
# 返回: 无
def seed_it(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置CNN_MNIST基准实验的所有必要组件
# 参数: num_client(客户端数), num_classes(分类数), epoch(训练轮数), device(设备), bs(批次大小), alpha(数据分布参数), lr(学习率), shuffle(是否打乱数据)
# 返回: (train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler)
def cnn_mnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, shuffle=False):
    train_loaders, validate_loader, complete_train_loader = mnist(num_client=num_client, bs=bs, alpha=alpha, shuffle=shuffle)
    global_model = cnn(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    schedulers = [None for i in optimizers]
    global_scheduler = None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def cnn_fashionmnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, shuffle=False):
    train_loaders, validate_loader, complete_train_loader = fashionmnist(num_client=num_client, bs=bs, alpha=alpha, shuffle=shuffle)
    global_model = cnn(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    schedulers = [None for i in optimizers]
    global_scheduler = None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

def alexnet_cifar10(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, shuffle=False, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = cifar10(num_client=num_client, bs=bs, alpha=alpha)
    global_model = AlexNet(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else False
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

def alexnet_cifar100(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, shuffle=False, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = cifar100(num_client=num_client, bs=bs, alpha=alpha)
    global_model = AlexNet(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else False
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

def alexnet_mnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, shuffle=False, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = mnist(
        num_client=num_client,
        bs=bs,
        alpha=alpha,
        shuffle=shuffle,
    )
    global_model = AlexNet_mnist(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else False
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def alexnet_fashionmnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, shuffle=False, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = fashionmnist(
        num_client=num_client,
        bs=bs,
        alpha=alpha,
        shuffle=shuffle,
        as_rgb=False,
        image_size=32,
    )
    global_model = AlexNet_mnist(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else False
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

# 设置VGG11_CIFAR10基准实验的所有必要组件
# 参数: num_client(客户端数), num_classes(分类数), epoch(训练轮数), device(设备), bs(批次大小), alpha(数据分布参数), lr(学习率)
# 返回: (train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler)
def vgg11_cifar10(num_client, num_classes, epoch, device, bs, alpha, lr=0.1):
    train_loaders, validate_loader, complete_train_loader = cifar10(num_client=num_client, bs=bs, alpha=alpha)
    global_model = vgg11(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)    
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch)
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def vgg16_cifar10(num_client, num_classes, epoch, device, bs, alpha, lr=0.1):
    train_loaders, validate_loader, complete_train_loader = cifar10(num_client=num_client, bs=bs, alpha=alpha)
    global_model = VGG16(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)    
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch)
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def vgg16_cifar100(num_client, num_classes, epoch, device, bs, alpha, lr=0.1):
    train_loaders, validate_loader, complete_train_loader = cifar100(num_client=num_client, bs=bs, alpha=alpha)
    global_model = VGG16(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch)
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

def vgg16_mnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1):
    train_loaders, validate_loader, complete_train_loader = mnist(
        num_client=num_client,
        bs=bs,
        alpha=alpha,
    )
    global_model = VGG16_mnist(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch)
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def vgg16_fashionmnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1):
    train_loaders, validate_loader, complete_train_loader = fashionmnist(
        num_client=num_client,
        bs=bs,
        alpha=alpha,
        as_rgb=False,
        image_size=32,
    )
    global_model = VGG16_mnist(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch)
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

# 设置VGG16_CIFAR10基准实验的所有必要组件
# 参数: num_client(客户端数), num_classes(分类数), epoch(训练轮数), device(设备), bs(批次大小), alpha(数据分布参数), lr(学习率)
# 返回: (train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler)# 设置ResNet18_CIFAR100基准实验的所有必要组件(使用SGD优化器)
# 参数: num_client(客户端数), num_classes(分类数), epoch(训练轮数), device(设备), bs(批次大小), alpha(数据分布参数), lr(学习率)
# 返回: (train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler)
def resnet18_cifar100(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = cifar100(num_client=num_client, bs=bs, alpha=alpha)
    global_model = Resnet18(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)    
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else True
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def resnet18_cifar10(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = cifar10(num_client=num_client, bs=bs, alpha=alpha)
    global_model = Resnet18(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else True
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

def resnet18_mnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = mnist(
        num_client=num_client,
        bs=bs,
        alpha=alpha,
    )
    global_model = Resnet18_mnist(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else True
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler


def resnet18_fashionmnist(num_client, num_classes, epoch, device, bs, alpha, lr=0.1, use_scheduler=None):
    train_loaders, validate_loader, complete_train_loader = fashionmnist(
        num_client=num_client,
        bs=bs,
        alpha=alpha,
        as_rgb=False,
        image_size=32,
    )
    global_model = Resnet18_mnist(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_enabled = bool(use_scheduler) if use_scheduler is not None else True
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers] if scheduler_enabled else [None for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch) if scheduler_enabled else None
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler

# 设置ResNet18_CIFAR100基准实验的所有必要组件(使用Adam优化器)
# 参数: num_client(客户端数), num_classes(分类数), epoch(训练轮数), device(设备), bs(批次大小), alpha(数据分布参数), lr(学习率)
# 返回: (train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler)
def resnet18_cifar100_adam(num_client, num_classes, epoch, device, bs, alpha, lr=0.1):
    train_loaders, validate_loader, complete_train_loader = cifar100(num_client=num_client, bs=bs, alpha=alpha)
    global_model = Resnet18(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.Adam(i.parameters(), lr=lr) for i in client_models]
    global_criterion = torch.nn.CrossEntropyLoss()
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)  
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=epoch)
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers, complete_train_loader, global_criterion, global_optimizer, global_scheduler