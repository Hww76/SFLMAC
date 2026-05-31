import torch
import torch.nn as nn

import timm

from torchvision.models import squeezenet1_1

# VGG网络结构配置字典
# 数字: 表示卷积层的输出通道数（filters数量）
# 'M': 表示MaxPooling层（最大池化层，kernel_size=2, stride=2）
# 'A': VGG11配置 - 8个卷积层 + 3个全连接层 = 11层
# 'B': VGG13配置 - 10个卷积层 + 3个全连接层 = 13层
# 'D': VGG16配置 - 13个卷积层 + 3个全连接层 = 16层
# 'E': VGG19配置 - 16个卷积层 + 3个全连接层 = 19层
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# 创建一个简单的CNN模型
# 参数: num_classes(分类数)
# 返回: CNN模型
def cnn(num_classes=10):
    model = CnnNet(num_classes=num_classes)
    return model

# 简单CNN网络，耄于MNIST数据集
# 参数: num_classes(分类数)
# 成员方法: forward(前向传播)
class CnnNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CnnNet, self).__init__()
        self.classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.advpool(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

# 创建一个VGG11模型
# 参数: num_classes(分类数)
# 返回: VGG11模型
def vgg11(num_classes=10):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_class=num_classes)

# 创建一个VGG16模型
# 参数: num_classes(分类数)
# 返回: VGG16模型
def vgg16(num_classes=10):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_classes)

# 创建一个ResNet18模型
# 参数: num_classes(分类数)
# 返回: ResNet18模型
def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# 创建一个Vision Transformer模型(MobileViT XXS)
# 参数: num_classes(分类数)
# 返回: Vision Transformer模型
def vit(num_classes=200):
    return timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)


# VGG模网络架构
# 参数: features(CNN特征提取层), num_class(分类数)
# 成员方法: forward(前向传播)
class VGG(nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

# 根据配置列表构建卷积层呫
# 参数: cfg(配置列表), batch_norm(是否使用批次正规化)
# 返回: 顺序网络层
def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l, track_running_stats=False)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)


# ResNet基本残差块
# 参数: in_channels(输入通道数), out_channels(输出通道数), stride(捷刖)
# 成员方法: forward(前向传播)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion, track_running_stats=False)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion, track_running_stats=False)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# ResNet模型架构
# 参数: block(残差块类型), num_block(每个殶中的残差块数), num_classes(分类数)
# 成员方法: forward(前向传播), _make_layer(构造殶)
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
