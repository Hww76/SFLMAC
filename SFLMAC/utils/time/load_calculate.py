# import sys
# from pathlib import Path

# REPO_ROOT = Path(__file__).resolve().parents[2]
# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))

from model.mnist.alexnet import AlexNet_mnist
from model.mnist.resnet18 import Resnet18_mnist
from model.mnist.vgg16 import VGG16_mnist

from model.alexnet import AlexNet
from model.vgg16 import VGG16
from model.resnet18 import Resnet18, BasicBlock
import torch
import torch.nn as nn


def calculate_conv_flops(in_channels, out_channels, kernel_size, input_h, input_w, output_h, output_w):
    forword_flops = 2 * in_channels * out_channels * kernel_size * kernel_size * output_h * output_w
    backword_flops = 2 * in_channels * out_channels * kernel_size * kernel_size * (input_h * input_w + output_h * output_w)
    return forword_flops , backword_flops

def calculate_maxpool_flops(out_channels, output_h, output_w):
    forword_flops = output_h * output_w * out_channels
    backword_flops = forword_flops
    return forword_flops , backword_flops

def calculate_linear_flops(input_channels, output_channels):
    forword_flops = (2*input_channels - 1) * output_channels
    backword_flops = (3*output_channels - 1) * input_channels
    return forword_flops , backword_flops

def calculate_ReLU_flops():
    return 0, 0


def _dataset_from_benchmark(benchmark: str) -> str:
    if 'cifar100' in benchmark:
        return 'cifar100'
    if 'cifar10' in benchmark:
        return 'cifar10'
    if 'fashionmnist' in benchmark:
        return 'fashionmnist'
    if 'mnist' in benchmark:
        return 'mnist'
    raise KeyError(f"Unsupported benchmark dataset in: {benchmark}")


def _resnet18_classic_layer_loads(batch_size: int, num_classes: int):
    forward_flops_list = [0]
    backward_flops_list = [0]
    parameter_bits_list = [0]
    activate_shape_list = [(batch_size, 3, 32, 32)]

    def append_conv_layer(in_c, out_c, k, in_h, in_w, out_h, out_w, extra_params=0):
        fwd, bwd = calculate_conv_flops(in_c, out_c, k, in_h, in_w, out_h, out_w)
        params = in_c * out_c * k * k + extra_params
        forward_flops_list.append(fwd)
        backward_flops_list.append(bwd)
        parameter_bits_list.append(params * 32)

    # layer1: stem conv (3->64, 32x32) + BN params
    append_conv_layer(3, 64, 3, 32, 32, 32, 32, extra_params=2 * 64)
    activate_shape_list.append((batch_size, 64, 32, 32))

    # 8 BasicBlocks, each counted as 2 conv layers (classic ResNet18 = 1 + 16 + 1)
    block_defs = [
        (64, 64, 1, False),
        (64, 64, 1, False),
        (64, 128, 2, True),
        (128, 128, 1, False),
        (128, 256, 2, True),
        (256, 256, 1, False),
        (256, 512, 2, True),
        (512, 512, 1, False),
    ]

    cur_h, cur_w = 32, 32
    for in_c, out_c, stride, has_shortcut in block_defs:
        # conv1 + bn1
        out_h = cur_h // stride
        out_w = cur_w // stride
        append_conv_layer(in_c, out_c, 3, cur_h, cur_w, out_h, out_w, extra_params=2 * out_c)
        activate_shape_list.append((batch_size, out_c, out_h, out_w))

        # conv2 + bn2 (+ shortcut conv/bn folded into this logical layer)
        extra_params = 2 * out_c
        if has_shortcut:
            shortcut_params = in_c * out_c * 1 * 1 + 2 * out_c
            extra_params += shortcut_params
        append_conv_layer(out_c, out_c, 3, out_h, out_w, out_h, out_w, extra_params=extra_params)
        activate_shape_list.append((batch_size, out_c, out_h, out_w))

        cur_h, cur_w = out_h, out_w

    # layer18: fc（严格使用 calculate_linear_flops）
    fwd, bwd = calculate_linear_flops(512, num_classes)
    fc_params = (512 * num_classes + num_classes) * 32
    forward_flops_list.append(fwd)
    backward_flops_list.append(bwd)
    parameter_bits_list.append(fc_params)
    activate_shape_list.append((batch_size, num_classes, 1, 1))

    return forward_flops_list, backward_flops_list, parameter_bits_list, activate_shape_list

def calculate_model_loads_by_layer(benchmark="alexnet_cifar10", batch_size=1):
    #/*
    #* 输入参数：model_name - 模型名称（字符串），如"AlexNet"
    #*          input_shape - 输入数据形状（元组），如(1,3,32,32)
    #* 输出参数：forward_flops_list - 每层前向FLOPs列表
    #*          backward_flops_list - 每层反向FLOPs列表
    #*          parameter_bits_list - 每层参数量（bits）列表
    #*/
    model = {
        'alexnet_cifar10': AlexNet(num_classes=10),
        'alexnet_cifar100': AlexNet(num_classes=100),
        'alexnet_mnist': AlexNet_mnist(num_classes=10),
        'alexnet_fashionmnist': AlexNet_mnist(num_classes=10),
        'vgg16_cifar10': VGG16(num_classes=10),
        'vgg16_cifar100': VGG16(num_classes=100),
        'vgg16_mnist': VGG16_mnist(num_classes=10),
        'vgg16_fashionmnist': VGG16_mnist(num_classes=10),
        'resnet18_cifar10': Resnet18(num_classes=10),
        'resnet18_mnist': Resnet18_mnist(num_classes=10),
        'resnet18_cifar100': Resnet18(num_classes=100),
        'resnet18_cifar100_adam': Resnet18(num_classes=100),
        'resnet18_fashionmnist': Resnet18_mnist(num_classes=10),
    }[benchmark]

    dataset = _dataset_from_benchmark(benchmark)
    input_shape = {
        'cifar10': (batch_size,3,32,32),
        'cifar100': (batch_size,3,32,32),
        'fashionmnist': (batch_size,1,32,32),
        'mnist': (batch_size,1,32,32),
    }[dataset]

    if benchmark.startswith('resnet18_'):
        num_classes = 100 if 'cifar100' in benchmark else 10
        return _resnet18_classic_layer_loads(batch_size=batch_size, num_classes=num_classes)

    input_height = input_shape[2]
    input_width = input_shape[3]
    forward_flops_list = [0]
    backward_flops_list = [0]
    parameter_bits_list = [0]
    activate_shape_list = []
    activate_shape_list.append(input_shape)

    def _layer_flops_bits_from_module(layer, h, w, c):
        forward_flops = 0
        backward_flops = 0
        parameter_bits = 0
        cur_h, cur_w, cur_c = h, w, c

        def _accumulate_from_module(module):
            nonlocal forward_flops, backward_flops, parameter_bits, cur_h, cur_w, cur_c
            if isinstance(module, nn.Conv2d):
                output_w = (cur_w - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
                output_h = (cur_h - module.kernel_size[1] + 2 * module.padding[1]) // module.stride[1] + 1
                tmp_f, tmp_b = calculate_conv_flops(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size[0],
                    cur_h,
                    cur_w,
                    output_h,
                    output_w,
                )
                forward_flops += tmp_f
                backward_flops += tmp_b
                parameter_bits += module.weight.numel()
                if module.bias is not None:
                    parameter_bits += module.bias.numel()
                cur_h, cur_w, cur_c = output_h, output_w, module.out_channels
            elif isinstance(module, nn.MaxPool2d):
                k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                s = module.stride if isinstance(module.stride, int) else module.stride[0]
                p = module.padding if isinstance(module.padding, int) else module.padding[0]
                output_w = (cur_w - k + 2 * p) // s + 1
                output_h = (cur_h - k + 2 * p) // s + 1
                tmp_f, tmp_b = calculate_maxpool_flops(cur_c, output_h, output_w)
                forward_flops += tmp_f
                backward_flops += tmp_b
                cur_h, cur_w = output_h, output_w
            elif isinstance(module, nn.Linear):
                tmp_f, tmp_b = calculate_linear_flops(module.in_features, module.out_features)
                forward_flops += tmp_f
                backward_flops += tmp_b
                parameter_bits += module.weight.numel()
                if module.bias is not None:
                    parameter_bits += module.bias.numel()
                cur_h, cur_w, cur_c = 1, 1, module.out_features
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    parameter_bits += module.weight.numel()
                if module.bias is not None:
                    parameter_bits += module.bias.numel()
            elif isinstance(module, nn.ReLU):
                tmp_f, tmp_b = calculate_ReLU_flops()
                forward_flops += tmp_f
                backward_flops += tmp_b

        if isinstance(layer, BasicBlock):
            identity_h, identity_w, identity_c = cur_h, cur_w, cur_c
            _accumulate_from_module(layer.conv1)
            _accumulate_from_module(layer.bn1)
            _accumulate_from_module(layer.relu)
            _accumulate_from_module(layer.conv2)
            _accumulate_from_module(layer.bn2)
            if len(layer.shortcut) > 0:
                sc_h, sc_w, sc_c = identity_h, identity_w, identity_c
                for sub in layer.shortcut:
                    if isinstance(sub, nn.Conv2d):
                        output_w = (sc_w - sub.kernel_size[0] + 2 * sub.padding[0]) // sub.stride[0] + 1
                        output_h = (sc_h - sub.kernel_size[1] + 2 * sub.padding[1]) // sub.stride[1] + 1
                        tmp_f, tmp_b = calculate_conv_flops(
                            sub.in_channels,
                            sub.out_channels,
                            sub.kernel_size[0],
                            sc_h,
                            sc_w,
                            output_h,
                            output_w,
                        )
                        forward_flops += tmp_f
                        backward_flops += tmp_b
                        parameter_bits += sub.weight.numel()
                        if sub.bias is not None:
                            parameter_bits += sub.bias.numel()
                        sc_h, sc_w, sc_c = output_h, output_w, sub.out_channels
                    elif isinstance(sub, nn.BatchNorm2d):
                        if sub.weight is not None:
                            parameter_bits += sub.weight.numel()
                        if sub.bias is not None:
                            parameter_bits += sub.bias.numel()
            _accumulate_from_module(layer.relu)
        else:
            modules = list(layer.children()) if len(list(layer.children())) > 0 else [layer]
            for sub in modules:
                _accumulate_from_module(sub)

        return forward_flops, backward_flops, parameter_bits, cur_h, cur_w, cur_c

    x = torch.zeros(*input_shape)
    if isinstance(model, Resnet18):
        layer_modules = [model.stem] + model._blocks() + [nn.Sequential(model.avg_pool, nn.Flatten(1), model.fc)]
    else:
        layer_modules = model.get_conv_layers_list() + model.get_fc_layers_list()

    for layer in layer_modules:
        forward_flops, backward_flops, parameter_bits, input_height, input_width, channels = _layer_flops_bits_from_module(
            layer, input_height, input_width, input_shape[1]
        )

        has_linear = any(isinstance(m, nn.Linear) for m in layer.modules())
        has_adaptive_pool = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in layer.modules())
        need_pre_flatten = isinstance(layer, nn.Linear) or (has_linear and not has_adaptive_pool)
        if need_pre_flatten and x.ndim > 2:
            x = torch.flatten(x, 1)

        with torch.no_grad():
            x = layer(x)
        if x.ndim == 4:
            input_shape = tuple(x.shape)
            channels = x.shape[1]
            input_height = x.shape[2]
            input_width = x.shape[3]
        else:
            input_shape = (x.shape[0], x.shape[1], 1, 1)
            channels = x.shape[1]
            input_height = 1
            input_width = 1

        activate_shape_list.append(input_shape)
        forward_flops_list.append(forward_flops)
        backward_flops_list.append(backward_flops)
        parameter_bits_list.append(parameter_bits)

    return forward_flops_list, backward_flops_list, [i*32 for i in parameter_bits_list], activate_shape_list

# 客户端总负载与传输/计算时间逻辑已迁移至 config/client.py 中的 Client_LBFSL。



if __name__ == "__main__":
    def build_cumulative_current_style(layer_loads: list):
        """复现当前 FSL/FSCL 中累计逻辑（含占位层偏移）。"""
        cumulative_self = [0]
        for i in range(1, len(layer_loads) + 1):
            cumulative_self.append(cumulative_self[i - 1] + layer_loads[i - 1])

        cumulative_helper = [0]
        for i in range(1, len(cumulative_self)):
            cumulative_helper.append(cumulative_self[-1] - cumulative_self[i])
        return cumulative_self, cumulative_helper

    def build_cumulative_corrected(layer_loads: list):
        """修正累计逻辑：忽略 layer_loads[0] 占位层，split_point=1 对应真实第一层。"""
        if len(layer_loads) == 0:
            return [0], [0]

        real_total = sum(layer_loads[1:])
        cumulative_self = [0]
        cumulative_helper = [0]
        running = 0
        for split_point in range(1, len(layer_loads)):
            running += layer_loads[split_point]
            cumulative_self.append(running)
            cumulative_helper.append(real_total - running)
        return cumulative_self, cumulative_helper

    def align_and_delta(current: list, corrected: list):
        max_len = max(len(current), len(corrected))
        cur = current + [None] * (max_len - len(current))
        cor = corrected + [None] * (max_len - len(corrected))
        delta = []
        for a, b in zip(cur, cor):
            if a is None or b is None:
                delta.append(None)
            else:
                delta.append(a - b)
        return cur, cor, delta

    benchmark_map = {
        "alexnet": "alexnet_cifar10",
        "resnet18": "resnet18_cifar10",
        "vgg16": "vgg16_cifar10",
    }

    output_path = REPO_ROOT / "load_calculate.log"
    lines = []

    for model_name, benchmark in benchmark_map.items():
        forward_flops_list, backward_flops_list, parameter_bits_list, activate_shape_list = calculate_model_loads_by_layer(
            benchmark=benchmark,
            batch_size=1,
        )

        f_self_cur, f_helper_cur = build_cumulative_current_style(forward_flops_list)
        b_self_cur, b_helper_cur = build_cumulative_current_style(backward_flops_list)
        p_self_cur, p_helper_cur = build_cumulative_current_style(parameter_bits_list)

        f_self_cor, f_helper_cor = build_cumulative_corrected(forward_flops_list)
        b_self_cor, b_helper_cor = build_cumulative_corrected(backward_flops_list)
        p_self_cor, p_helper_cor = build_cumulative_corrected(parameter_bits_list)

        f_self_cur, f_self_cor, f_self_delta = align_and_delta(f_self_cur, f_self_cor)
        f_helper_cur, f_helper_cor, f_helper_delta = align_and_delta(f_helper_cur, f_helper_cor)
        b_self_cur, b_self_cor, b_self_delta = align_and_delta(b_self_cur, b_self_cor)
        b_helper_cur, b_helper_cor, b_helper_delta = align_and_delta(b_helper_cur, b_helper_cor)
        p_self_cur, p_self_cor, p_self_delta = align_and_delta(p_self_cur, p_self_cor)
        p_helper_cur, p_helper_cor, p_helper_delta = align_and_delta(p_helper_cur, p_helper_cor)

        lines.append("=" * 90)
        lines.append(f"model={model_name}, benchmark={benchmark}")
        lines.append("-" * 90)

        lines.append("[原始层负载列表]")
        lines.append(f"forward_flops_list={forward_flops_list}")
        lines.append(f"backward_flops_list={backward_flops_list}")
        lines.append(f"parameter_bits_list={parameter_bits_list}")
        lines.append("")

        lines.append("[当前FSL/FSCL累计逻辑 vs 修正逻辑 对比]")
        lines.append(
            "split_point | "
            "f_self(cur/cor/delta) | f_helper(cur/cor/delta) | "
            "b_self(cur/cor/delta) | b_helper(cur/cor/delta) | "
            "p_self(cur/cor/delta) | p_helper(cur/cor/delta)"
        )

        max_len = max(
            len(f_self_cur), len(f_helper_cur), len(b_self_cur), len(b_helper_cur), len(p_self_cur), len(p_helper_cur)
        )

        def _fmt(v):
            return "NA" if v is None else str(v)

        for split_point in range(0, max_len):
            lines.append(
                f"{split_point:>11} | "
                f"{_fmt(f_self_cur[split_point])}/{_fmt(f_self_cor[split_point])}/{_fmt(f_self_delta[split_point])} | "
                f"{_fmt(f_helper_cur[split_point])}/{_fmt(f_helper_cor[split_point])}/{_fmt(f_helper_delta[split_point])} | "
                f"{_fmt(b_self_cur[split_point])}/{_fmt(b_self_cor[split_point])}/{_fmt(b_self_delta[split_point])} | "
                f"{_fmt(b_helper_cur[split_point])}/{_fmt(b_helper_cor[split_point])}/{_fmt(b_helper_delta[split_point])} | "
                f"{_fmt(p_self_cur[split_point])}/{_fmt(p_self_cor[split_point])}/{_fmt(p_self_delta[split_point])} | "
                f"{_fmt(p_helper_cur[split_point])}/{_fmt(p_helper_cor[split_point])}/{_fmt(p_helper_delta[split_point])}"
            )

        lines.append("-" * 90)
        lines.append(f"total_forward(real)={sum(forward_flops_list[1:])}")
        lines.append(f"total_backward(real)={sum(backward_flops_list[1:])}")
        lines.append(f"total_bits(real)={sum(parameter_bits_list[1:])}")
        lines.append(f"activate_shape_count={len(activate_shape_list)}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved cumulative load report to: {output_path}")