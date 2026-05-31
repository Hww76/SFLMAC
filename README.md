# SFLMAC Quick Start Guide

This project currently supports the following algorithms:

- `FedAvg`
- `SplitFed`
- `RingSFL`
- `SFLMAC`

## 1. Environment Setup
```bash
pip install -r requirements.txt
```

If you use Conda:

```bash
conda create -n sflmac python=3.11 -y
conda activate sflmac
pip install -r requirements.txt
```

## 2. Running Guide

```bash
python main.py \
	--benchmark resnet18_cifar10 \
	--num_client 8 \
	--global_epoch 5 \
	--local_epoch 2 \
	--bs 64 \
	--lr 0.01 \
	--alpha 5 \
	--device cuda:0 \
	--output SFLMAC_quickstart
```

Notes:

- The prefix before the first underscore in `--output` is used to identify the algorithm.
- For example, `SFLMAC_quickstart` automatically selects the `SFLMAC` algorithm.
- If `--output` is not passed, results will land in a default directory — it is recommended to always provide it.

## 3. Minimal Examples for All Four Algorithms

Simply change the `--output` prefix to switch algorithms:

```bash
# FedAvg
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --output FedAvg_demo

# SplitFed (optional: --split_point)
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --split_point 2 --output SplitFed_demo

# RingSFL (optional: --ring_size)
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --ring_size 10 --output RingSFL_demo

# SFLMAC (optional: --split_point)
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --split_point 2 --output SFLMAC_demo
```

## 4. Output Location

When `--output` is provided, results are saved to:

`results/<distribution_dir>/<benchmark>/global_epoch{G}_local_epoch{L}_num_client{N}/<Algorithm>/`

Rules:

- Distribution directory:
	- `IID_alpha5` (when `alpha == 5`)
	- `NonIID_alpha{alpha}` (when `alpha != 5`)
- `<Algorithm>` is identified from the `--output` prefix, e.g., `SFLMAC_demo` → `SFLMAC`

Typical output files:

- `<prefix>.log` (e.g., `SFLMAC.log` or `FedAvg.log`)
- `loss.npy`
- `acc1.npy`
- `acc5.npy`
- `time.npy`
- `clients_time.npy`

## 5. Common Parameters Quick Reference

- `--benchmark`: Model + dataset combination (e.g., `resnet18_cifar10`)
- `--num_client`: Number of clients
- `--global_epoch`: Global rounds
- `--local_epoch`: Local rounds
- `--bs`: Batch size
- `--lr`: Learning rate
- `--alpha`: Data distribution parameter (5 means IID)
- `--device`: Training device (e.g., `cuda:0` or `cpu`)
- `--split_point`: Commonly used by `SplitFed` / `SFLMAC`
- `--ring_size`: Commonly used by `RingSFL`

## 6. Quick Troubleshooting

- `KeyError` or algorithm mismatch: Verify that the `--output` prefix is one of `FedAvg`/`SplitFed`/`RingSFL`/`SFLMAC`.
- Out of memory (OOM): Reduce `--bs`, or test the pipeline with `--device cpu` first.
- Mismatched number of classes: `main.py` automatically corrects `num_classes` based on the `benchmark` — manual adjustment is generally not needed.

## 7. Results

Our model achieves the following performance on CIFAR-10:

where N is the number of clients.

Model: AlexNet

| Method           |    N=8   |   N=16   |   N=32   |
| ---------------- | -------: | -------: | -------: |
| FedAvg           |   69.99  |   70.53  |   70.91  |
| SplitFed         |   78.04  |   79.32  |   80.28  |
| RingSFL          |   69.43  |   70.70  |   71.27  |
| SFLMAC (Ours)    |   78.97  |   80.31  |   80.49  |

Model: VGG16

| Method           |    N=8   |   N=16   |   N=32   |
| ---------------- | -------: | -------: | -------: |
| FedAvg           |   84.51  |   84.75  |   85.20  |
| SplitFed         |   87.00  |   86.68  |   86.58  |
| RingSFL          |   85.93  |   86.85  |   86.76  |
| SFLMAC (Ours)    |   87.84  |   88.88  |   89.08  |

Model: ResNet18

| Method           |    N=8   |   N=16   |   N=32   |
| ---------------- | -------: | -------: | -------: |
| FedAvg           |   83.94  |   84.91  |   85.10  |
| SplitFed         |   88.13  |   88.02  |   86.85  |
| RingSFL          |   86.63  |   87.14  |   86.80  |
| SFLMAC (Ours)    |   88.67  |   89.53  |   89.79  |
