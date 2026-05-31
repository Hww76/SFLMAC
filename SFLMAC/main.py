import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import numpy as np
import sys
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import redirect_stdout
from typing import Any

from Algorithm.FedAvg.core import FedAvg
from Algorithm.RingSFL.core import RingSFL
from Algorithm.SplitFed.core import SplitFed
from Algorithm.SFLMAC.core import SFLMAC

from utils.choose_model_dataset import alexnet_cifar10, alexnet_cifar100, seed_it, vgg11_cifar10, vgg16_cifar10, vgg16_cifar100, resnet18_cifar10, resnet18_cifar100, resnet18_cifar100_adam, cnn_fashionmnist, alexnet_fashionmnist, vgg16_fashionmnist, resnet18_fashionmnist, alexnet_mnist, vgg16_mnist, resnet18_mnist
# 设置随机种子保证实验可重复性
seed_it(42)

# 创建命令行参数解析器，用于配置实验参数
# 基准(--benchmark): 选择数据集和模型组合
# 分类数(--num_classes): 分类任务的类别数
# 批次大小(--bs): 训练批次大小
# 学习率(--lr): 初始学习率
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, choices=['alexnet_cifar10', 'alexnet_cifar100', 'vgg11_cifar10', 'vgg16_cifar10', 'vgg16_cifar100', 'resnet18_cifar10', 'resnet18_cifar100', 'resnet18_cifar100_adam', 'cnn_fashionmnist', 'alexnet_fashionmnist', 'vgg16_fashionmnist', 'resnet18_fashionmnist', 'alexnet_mnist', 'vgg16_mnist', 'resnet18_mnist'], default='vgg16_cifar10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)

parser.add_argument('--num_client', type=int, default=10)
parser.add_argument('--alpha', type=float, default=5)
parser.add_argument('--global_epoch', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--beta', type=float, default=0.1)

parser.add_argument('--device', default='cuda:1')

parser.add_argument('--output', type=str)
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--use_scheduler', type=int, choices=[-1, 0, 1], default=-1)

# SplitFed/SFLMAC shared split point
parser.add_argument('--split_point', type=int, default=None)

# RingSFL 特有参数
parser.add_argument('--ring_size', type=int, default=16.)

arg = parser.parse_args()

def normalize_num_classes(benchmark: str, num_classes: int) -> int:
    if 'cifar100' in benchmark:
        if num_classes != 100:
            print(f"[WARN] benchmark={benchmark} 需要 num_classes=100，已从 {num_classes} 自动调整为 100", file=sys.__stdout__)
        return 100
    if 'cifar10' in benchmark:
        if num_classes != 10:
            print(f"[WARN] benchmark={benchmark} 需要 num_classes=10，已从 {num_classes} 自动调整为 10", file=sys.__stdout__)
        return 10
    if 'mnist' in benchmark:
        if num_classes != 10:
            print(f"[WARN] benchmark={benchmark} 需要 num_classes=10，已从 {num_classes} 自动调整为 10", file=sys.__stdout__)
        return 10
    return num_classes


arg.num_classes = normalize_num_classes(arg.benchmark, arg.num_classes)


@dataclass
class TrainContext:
    global_model: Any
    client_models: Any
    criterions: Any
    optimizers: Any
    schedulers: Any
    global_criterion: Any
    global_optimizer: Any
    global_scheduler: Any
    dataloaders: Any
    valloader: Any
    completeloader: Any
    device: str
    global_epoch: int
    local_epoch: int
    beta: float
    benchmark: str
    bs: int
    alpha: float
    debug: bool
    private_data: dict = field(default_factory=dict)

benchmark_factory = {
    'alexnet_cifar10': alexnet_cifar10,
    'alexnet_cifar100': alexnet_cifar100,
    'vgg16_cifar10': vgg16_cifar10,
    'vgg16_cifar100': vgg16_cifar100,
    'vgg11_cifar10': vgg11_cifar10,
    'resnet18_cifar10': resnet18_cifar10,
    'resnet18_cifar100': resnet18_cifar100,
    'resnet18_cifar100_adam': resnet18_cifar100_adam,
    'cnn_fashionmnist': cnn_fashionmnist,
    'alexnet_mnist': alexnet_mnist,
    'vgg16_mnist': vgg16_mnist,
    'resnet18_mnist': resnet18_mnist,
    'alexnet_fashionmnist': alexnet_fashionmnist,
    'vgg16_fashionmnist': vgg16_fashionmnist,
    'resnet18_fashionmnist': resnet18_fashionmnist,
}[arg.benchmark]

factory_kwargs = dict(
    num_client=arg.num_client,
    num_classes=arg.num_classes,
    epoch=arg.global_epoch * arg.local_epoch,
    device=arg.device,
    bs=arg.bs,
    alpha=arg.alpha,
    lr=arg.lr,
)
if "use_scheduler" in inspect.signature(benchmark_factory).parameters:
    factory_kwargs["use_scheduler"] = None if arg.use_scheduler < 0 else bool(arg.use_scheduler)

train_loaders, validate_loader, global_model, client_models, \
criterions, optimizers, schedulers, complete_train_loader, \
global_criterion, global_optimizer, global_scheduler = benchmark_factory(**factory_kwargs)

output_name = arg.output or "no"
output_parts = output_name.split("_")
log_name = output_parts[0] if arg.output else "train"

algo_alias = {
    'fedavg': 'FedAvg',
    'ringsfl': 'RingSFL',
    'splitfed': 'SplitFed',
    'sflmac': 'SFLMAC',
}
algo_key = algo_alias.get(output_parts[0].lower(), output_parts[0])
stabilize_method = None


algorithm = {
    'FedAvg': FedAvg,
    'RingSFL': RingSFL,
    'SplitFed': SplitFed,
    'SFLMAC': SFLMAC,
}[algo_key]

results_root = Path("results")
save_dir = results_root
run_dir_name = (
    f"global_epoch{arg.global_epoch}_"
    f"local_epoch{arg.local_epoch}_"
    f"num_client{arg.num_client}"
)

if arg.output:
    alpha_value = float(arg.alpha)
    alpha_str = f"{alpha_value:g}"
    distribution_dir = f"IID_alpha{alpha_str}" if np.isclose(alpha_value, 5.0) else f"NonIID_alpha{alpha_str}"
    save_dir = results_root / distribution_dir / arg.benchmark / run_dir_name / algo_key

save_dir.mkdir(parents=True, exist_ok=True)

# 创建联邦学习上下文并初始化算法实例
train_ctx = TrainContext(
    global_model=global_model,
    client_models=client_models,
    criterions=criterions,
    optimizers=optimizers,
    schedulers=schedulers,
    global_criterion=global_criterion,
    global_optimizer=global_optimizer,
    global_scheduler=global_scheduler,
    dataloaders=train_loaders,
    valloader=validate_loader,
    completeloader=complete_train_loader,
    device=arg.device,
    global_epoch=arg.global_epoch,
    local_epoch=arg.local_epoch,
    beta=arg.beta,
    benchmark=arg.benchmark,
    bs=arg.bs,
    alpha=arg.alpha,
    debug=arg.debug,
    private_data={
        "stabilize_method": stabilize_method,
        'SFLMAC': {
            'split_point': arg.split_point,
        },
        'RingSFL': {
            'ring_size': arg.ring_size,
        },
        'SplitFed': {
            'split_point': arg.split_point,
        },
    },
)

print(f"save_dir: {save_dir}", file=sys.__stdout__)

# 执行联邦学习训练过程，返回每个全局轮的损失和准确率
with open(save_dir / f"{log_name}.log", "w") as _log_f:
    with redirect_stdout(_log_f):
        if arg.debug:
            print("[Debug][HyperParams] ====================================")
            print(f"[Debug][HyperParams] algorithm={algo_key}")
            print(f"[Debug][HyperParams] benchmark={arg.benchmark}")
            print(f"[Debug][HyperParams] num_classes={arg.num_classes}")
            print(f"[Debug][HyperParams] num_client={arg.num_client}")
            print(f"[Debug][HyperParams] global_epoch={arg.global_epoch}")
            print(f"[Debug][HyperParams] local_epoch={arg.local_epoch}")
            print(f"[Debug][HyperParams] bs={arg.bs}")
            print(f"[Debug][HyperParams] lr={arg.lr}")
            print(f"[Debug][HyperParams] alpha={arg.alpha}")
            print(f"[Debug][HyperParams] beta={arg.beta}")
            print(f"[Debug][HyperParams] split_point={arg.split_point}")
            print(f"[Debug][HyperParams] device={arg.device}")
            print(f"[Debug][HyperParams] output={arg.output}")
            print(f"[Debug][HyperParams] ring_size={arg.ring_size}")
            print(f"[Debug][HyperParams] use_scheduler={arg.use_scheduler}")
            print(f"[Debug][HyperParams] save_dir={save_dir}")
            print("[Debug][HyperParams] ====================================")
        fl = algorithm(train_ctx)
        loss_lst, acc1_lst, acc5_lst, time_lst, clients_time_lst = fl.run()
        best_acc1 = max(acc1_lst) if len(acc1_lst) > 0 else 0.0
        print(f"[Summary] best_acc1={best_acc1:.6f}")

comm_stats = fl.get_communication_stats() if hasattr(fl, "get_communication_stats") else None
if comm_stats is not None:
    np.save(save_dir / 'comm_model_bits.npy', np.array(comm_stats.get("round_model_bits", []), dtype=np.float64))
    np.save(save_dir / 'comm_activation_bits.npy', np.array(comm_stats.get("round_activation_bits", []), dtype=np.float64))
    np.save(save_dir / 'comm_total_bits.npy', np.array([
        comm_stats.get("total_model_bits", 0),
        comm_stats.get("total_activation_bits", 0),
        comm_stats.get("total_bits", 0),
        comm_stats.get("total_bytes", 0.0),
    ], dtype=np.float64))

    communication_dir = results_root / "communication" / algo_key / arg.benchmark / run_dir_name
    communication_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "algorithm": algo_key,
        "benchmark": arg.benchmark,
        "global_epoch": arg.global_epoch,
        "local_epoch": arg.local_epoch,
        "num_client": arg.num_client,
        "model_bits": comm_stats.get("total_model_bits", 0),
        "model_bits_by_role": comm_stats.get("total_model_bits_by_role", {}),
        "activation_bits": comm_stats.get("total_activation_bits", 0),
        "total_bits": comm_stats.get("total_bits", 0),
        "total_bytes": comm_stats.get("total_bytes", 0.0),
    }
    with open(communication_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    with open(communication_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"algorithm={summary_payload['algorithm']}\n")
        f.write(f"benchmark={summary_payload['benchmark']}\n")
        f.write(f"global_epoch={summary_payload['global_epoch']}\n")
        f.write(f"local_epoch={summary_payload['local_epoch']}\n")
        f.write(f"num_client={summary_payload['num_client']}\n")
        f.write(f"model_bits={summary_payload['model_bits']}\n")
        f.write(f"model_bits_by_role={summary_payload['model_bits_by_role']}\n")
        f.write(f"activation_bits={summary_payload['activation_bits']}\n")
        f.write(f"total_bits={summary_payload['total_bits']}\n")
        f.write(f"total_bytes={summary_payload['total_bytes']}\n")

# 如果指定了输出文件名，则保存训练结果到results文件夹
if arg.output != 'no':
    np.save(save_dir / f'loss.npy', np.array(loss_lst))
    np.save(save_dir / f'acc1.npy', np.array(acc1_lst))
    np.save(save_dir / f'acc5.npy', np.array(acc5_lst))
    np.save(save_dir / f'time.npy', np.array(time_lst))
    np.save(save_dir / f'clients_time.npy', np.array(clients_time_lst))