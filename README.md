# SFLMAC 快速启动指南

本项目当前保留并支持以下算法：

- `FedAvg`
- `SplitFed`
- `RingSFL`
- `SFLMAC`

## 1. 环境准备
```bash
pip install -r requirements.txt
```

如果你使用 Conda：

```bash
conda create -n sflmac python=3.11 -y
conda activate sflmac
pip install -r requirements.txt
```

## 2. 运行指南

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

说明：

- `--output` 第一个下划线前的前缀会作为算法名识别。
- 例如 `SFLMAC_quickstart` 会自动选择 `SFLMAC` 算法。
- 如果没有传 `--output`，结果会落在默认目录，建议始终传入。

## 3. 四种算法最小示例

只需替换 `--output` 前缀即可切换算法：

```bash
# FedAvg
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --output FedAvg_demo

# SplitFed（可加 --split_point）
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --split_point 2 --output SplitFed_demo

# RingSFL（可加 --ring_size）
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --ring_size 10 --output RingSFL_demo

# SFLMAC（可加 --split_point）
python main.py --benchmark resnet18_cifar10 --num_client 8 --global_epoch 5 --local_epoch 2 --bs 64 --lr 0.01 --alpha 5 --device cuda:0 --split_point 2 --output SFLMAC_demo
```

## 4. 结果保存位置

当传入 `--output` 时，结果会保存到：

`results/<分布目录>/<benchmark>/global_epoch{G}_local_epoch{L}_num_client{N}/<Algorithm>/`

规则：

- 分布目录：
	- `IID_alpha5`（当 `alpha == 5`）
	- `NonIID_alpha{alpha}`（当 `alpha != 5`）
- `<Algorithm>` 由 `--output` 前缀识别，例如 `SFLMAC_demo` -> `SFLMAC`

典型输出文件：

- `<prefix>.log`（例如 `SFLMAC.log` 或 `FedAvg.log`）
- `loss.npy`
- `acc1.npy`
- `acc5.npy`
- `time.npy`
- `clients_time.npy`

## 5. 常用参数速查

- `--benchmark`：模型+数据集组合（如 `resnet18_cifar10`）
- `--num_client`：客户端数量
- `--global_epoch`：全局轮次
- `--local_epoch`：本地轮次
- `--bs`：batch size
- `--lr`：学习率
- `--alpha`：数据分布参数（5 代表 IID）
- `--device`：训练设备（如 `cuda:0` 或 `cpu`）
- `--split_point`：`SplitFed`/`SFLMAC` 常用
- `--ring_size`：`RingSFL` 常用

## 6. 快速排错

- 报错 `KeyError` 或算法不匹配：检查 `--output` 前缀是否是 `FedAvg/SplitFed/RingSFL/SFLMAC`。
- 显存不足：减小 `--bs`，或先用 `--device cpu` 验证流程。
- 类别数不一致：`main.py` 会按 `benchmark` 自动修正 `num_classes`，一般不需要手改。
