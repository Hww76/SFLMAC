import torch
import torch.nn as nn

from config import training as training_config


def get_components(num_classes: int):
    conv_components = [
        nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64, track_running_stats=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)),
        nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128, track_running_stats=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)),
        nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256, track_running_stats=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)),
        nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, track_running_stats=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)),
        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, track_running_stats=False), nn.ReLU(inplace=True)),
        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, track_running_stats=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)),
    ]

    fc_components = [
        nn.Sequential(nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)),
        nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)),
        nn.Linear(4096, num_classes),
    ]
    return conv_components, fc_components


def build_model_by_split_point(num_classes: int, split_point: int):
    if not isinstance(split_point, int) or split_point < 1:
        raise ValueError("split_point必须是≥1的整数")

    conv_components, fc_components = get_components(num_classes)
    selected_conv_components = []
    selected_fc_components = []

    conv_limit = min(split_point, len(conv_components))
    for idx in range(conv_limit):
        selected_conv_components.append(conv_components[idx])

    if split_point < len(conv_components):
        return nn.Sequential(*selected_conv_components), None

    fc_end_idx = min(split_point - len(conv_components), len(fc_components))
    for idx in range(fc_end_idx):
        selected_fc_components.append(fc_components[idx])
    return nn.Sequential(*selected_conv_components), nn.Sequential(*selected_fc_components)


class VGG16_mnist(nn.Module):
    def __init__(self, num_classes=10, split_point=training_config.VGG16_MAX_SPLIT_POINT):
        super().__init__()
        self.max_layer_len = training_config.VGG16_MAX_SPLIT_POINT
        self.max_conv_layers = 13
        self.max_fc_layers = 3

        if split_point > self.max_layer_len or split_point < 1:
            raise ValueError(f"分割点超出范围！请设置在1-{self.max_layer_len}之间。")

        self.num_class = num_classes
        self.split_point = split_point
        self.conv, self.fc = build_model_by_split_point(num_classes, split_point)

    def forward(self, x, split_point=None, lower=True, return_features=False):
        if split_point is None:
            split_point = self.split_point

        if split_point == training_config.VGG16_MAX_SPLIT_POINT:
            out = self.conv(x)
            features = out.view(out.size(0), -1)
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

    def lower_forward(self, x, split_point=training_config.VGG16_MAX_SPLIT_POINT, return_features=False):
        out = x
        features = None

        if split_point < self.max_conv_layers:
            conv_layers = self.get_conv_layers_list()
            for idx in range(split_point):
                out = conv_layers[idx](out)
        elif split_point < self.max_conv_layers + self.max_fc_layers:
            out = self.conv(out)
            features = out.view(out.size(0), -1)
            out = features
            fc_start_idx = split_point - self.max_conv_layers
            fc_layers = self.get_fc_layers_list()
            for idx in range(fc_start_idx):
                out = fc_layers[idx](out)
        else:
            out = self.conv(out)
            features = out.view(out.size(0), -1)
            out = self.fc(features)

        if return_features:
            return out, features
        return out

    def upper_forward(self, x, split_point=training_config.VGG16_MAX_SPLIT_POINT, return_features=False):
        out = x
        features = None

        if split_point < self.max_conv_layers:
            conv_layers = self.get_conv_layers_list()
            for idx in range(split_point, self.max_conv_layers):
                out = conv_layers[idx](out)
            features = out.view(out.size(0), -1)
            out = self.fc(features)
        elif split_point < self.max_conv_layers + self.max_fc_layers:
            fc_start_idx = split_point - self.max_conv_layers
            fc_layers = self.get_fc_layers_list()
            for idx in range(fc_start_idx, self.max_fc_layers):
                out = fc_layers[idx](out)
        else:
            out = self.conv(out)
            features = out.view(out.size(0), -1)
            out = self.fc(features)

        if return_features:
            return out, features
        return out

    def get_conv_layers_list(self):
        return list(self.conv.children())

    def get_fc_layers_list(self):
        return list(self.fc.children())


if __name__ == "__main__":
    model = VGG16_mnist(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)