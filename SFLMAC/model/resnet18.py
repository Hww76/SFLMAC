import torch
import torch.nn as nn

from config import training as training_config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion, track_running_stats=False),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(identity)
        out = self.relu(out)
        return out


class _ParamLayer:
    def __init__(self, named_params=None, named_buffers=None):
        self._named_params = list(named_params or [])
        self._named_buffers = list(named_buffers or [])

    def parameters(self):
        return (param for _, param in self._named_params)

    def named_parameters(self, recurse=True):
        return iter(self._named_params)

    def named_buffers(self, recurse=True):
        return iter(self._named_buffers)


class Resnet18(nn.Module):
    def __init__(self, num_classes=100, split_point=training_config.RESNET18_MAX_SPLIT_POINT):
        super().__init__()

        self.max_layer_len = training_config.RESNET18_MAX_SPLIT_POINT
        self.max_conv_layers = 17
        self.max_fc_layers = 1

        if split_point > self.max_layer_len or split_point < 1:
            raise ValueError(f"分割点超出范围！请设置在1-{self.max_layer_len}之间。")

        self.split_point = split_point

        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for cur_stride in strides:
            blocks.append(BasicBlock(self.in_channels, out_channels, stride=cur_stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _blocks(self):
        return [
            self.layer1[0], self.layer1[1],
            self.layer2[0], self.layer2[1],
            self.layer3[0], self.layer3[1],
            self.layer4[0], self.layer4[1],
        ]

    def _executed_blocks(self, split_point: int) -> int:
        if split_point <= 1:
            return 0
        return min(8, (split_point - 1 + 1) // 2)

    def forward(self, x, split_point=None, lower=True, return_features=False):
        if split_point is None:
            split_point = self.split_point

        if split_point == training_config.RESNET18_MAX_SPLIT_POINT:
            out = self.stem(x)
            for block in self._blocks():
                out = block(out)
            features = self.avg_pool(out)
            features = torch.flatten(features, 1)
            out = self.fc(features)
        else:
            if lower:
                if return_features:
                    out, features = self.lower_forward(x=x, split_point=split_point, return_features=return_features)
                else:
                    out = self.lower_forward(x=x, split_point=split_point, return_features=return_features)
            else:
                if return_features:
                    out, features = self.upper_forward(x=x, split_point=split_point, return_features=return_features)
                else:
                    out = self.upper_forward(x=x, split_point=split_point, return_features=return_features)

        if return_features:
            return out, features
        return out

    def lower_forward(self, x, split_point=training_config.RESNET18_MAX_SPLIT_POINT, return_features=False):
        out = x
        features = None

        if split_point >= 1:
            out = self.stem(out)

        executed_blocks = self._executed_blocks(split_point)
        blocks = self._blocks()
        for idx in range(executed_blocks):
            out = blocks[idx](out)

        if split_point >= self.max_layer_len:
            features = self.avg_pool(out)
            features = torch.flatten(features, 1)
            out = self.fc(features)

        if return_features:
            return out, features
        return out

    def upper_forward(self, x, split_point=training_config.RESNET18_MAX_SPLIT_POINT, return_features=False):
        out = x
        features = None

        if split_point >= self.max_layer_len:
            out = self.stem(out)
            for block in self._blocks():
                out = block(out)
            features = self.avg_pool(out)
            features = torch.flatten(features, 1)
            out = self.fc(features)
            if return_features:
                return out, features
            return out

        blocks = self._blocks()
        executed_blocks = self._executed_blocks(split_point)
        for idx in range(executed_blocks, len(blocks)):
            out = blocks[idx](out)

        features = self.avg_pool(out)
        features = torch.flatten(features, 1)
        out = self.fc(features)

        if return_features:
            return out, features
        return out

    def get_conv_layers_list(self):
        param_name_by_id = {id(p): name for name, p in self.named_parameters()}
        buffer_name_by_id = {id(b): name for name, b in self.named_buffers()}

        def collect_named_tensors(modules):
            named_params = []
            named_buffers = []
            for module in modules:
                for param in module.parameters(recurse=True):
                    name = param_name_by_id.get(id(param))
                    if name is not None:
                        named_params.append((name, param))
                for buffer in module.buffers(recurse=True):
                    name = buffer_name_by_id.get(id(buffer))
                    if name is not None:
                        named_buffers.append((name, buffer))
            return named_params, named_buffers

        stem_named_params, stem_named_buffers = collect_named_tensors([self.stem[0], self.stem[1]])
        conv_layers = [_ParamLayer(stem_named_params, stem_named_buffers)]

        for block in self._blocks():
            layer1_params, layer1_buffers = collect_named_tensors([block.conv1, block.bn1])

            layer2_modules = [block.conv2, block.bn2]
            if len(block.shortcut) > 0:
                for module in block.shortcut:
                    layer2_modules.append(module)
            layer2_params, layer2_buffers = collect_named_tensors(layer2_modules)

            conv_layers.append(_ParamLayer(layer1_params, layer1_buffers))
            conv_layers.append(_ParamLayer(layer2_params, layer2_buffers))
        return conv_layers

    def get_fc_layers_list(self):
        param_name_by_id = {id(p): name for name, p in self.named_parameters()}
        buffer_name_by_id = {id(b): name for name, b in self.named_buffers()}
        fc_named_params = []
        fc_named_buffers = []
        for param in self.fc.parameters(recurse=True):
            name = param_name_by_id.get(id(param))
            if name is not None:
                fc_named_params.append((name, param))
        for buffer in self.fc.buffers(recurse=True):
            name = buffer_name_by_id.get(id(buffer))
            if name is not None:
                fc_named_buffers.append((name, buffer))
        return [_ParamLayer(fc_named_params, fc_named_buffers)]


if __name__ == "__main__":
    model = Resnet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)