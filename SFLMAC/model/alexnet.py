import torch
import torch.nn as nn
from config import training as training_config

def get_components(num_classes):
    """
    拆解原始卷积结构为按顺序的组件列表（每个组件对应1个Conv2d层）
    返回：组件列表，每个元素是原始结构中的一层/嵌套Sequential
    """
    conv_components = [
        # 组件0：对应第1个Conv2d（标注0）
        nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ),
        # 组件1：对应第2个Conv2d（标注1）
        nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ),
        # 组件2：对应第3个Conv2d（标注2）
        nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ),
        # 组件3：对应第4个Conv2d（标注3）
        nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ),
        # 组件4：对应第5个Conv2d（标注4）
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    ] ## 分割点为5

    fc_components = [
        nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),                    # 0
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),                            # 1
            nn.ReLU(inplace=True),
        ),
        nn.Linear(512, num_classes),                        # 2
    ]

    return conv_components, fc_components

def build_model_by_split_point(num_classes, split_point):
    """
    根据split_point生成卷积模型，确保模型中Conv2d层数 < split_point
    :param split_point: 分割点（整数），需 ≥1（否则无有效Conv2d层）
    :return: 
        model: 生成的nn.Sequential模型
        actual_conv_num: 模型中实际的Conv2d层数
    """
    # 合法性校验
    if not isinstance(split_point, int) or split_point < 1:
        raise ValueError("split_point必须是≥1的整数")
    
    conv_components, fc_components = get_components(num_classes)
    actual_conv_num = 0  # 累计Conv2d层数
    selected_conv_components = []  # 选中的组件（用于构建模型）
    selected_fc_components = []    # 选中的fc组件（若需要）

    # 遍历组件，累计Conv2d层数，直到接近但不超过split_point-1
    for comp in conv_components:
        # 每个组件固定对应1个Conv2d，判断是否满足 "累计数+1 < split_point"
        if actual_conv_num < split_point:
            selected_conv_components.append(comp)
            actual_conv_num += 1
        else:
            break  # 停止添加，避免层数超过限制
    if split_point < 5:
        return nn.Sequential(*selected_conv_components), None
    else: 
        # 分割点跨越卷积层和全连接层
        fc_end_idx = split_point - 5  # 计算需要添加的fc层数
        for idx in range(fc_end_idx):
            selected_fc_components.append(fc_components[idx])
        return nn.Sequential(*selected_conv_components), nn.Sequential(*selected_fc_components)

class AlexNet(nn.Module):
    """基类：仅定义完整AlexNet的层结构 + 完整模型的前向执行策略"""
    def __init__(self, num_classes=10, split_point=training_config.ALEXNET_MAX_SPLIT_POINT):
        super(AlexNet, self).__init__()

        self.max_layer_len = training_config.ALEXNET_MAX_SPLIT_POINT
        self.now_layer_len = 0
        self.max_conv_layers = 5
        self.max_fc_layers = 3
        
        if split_point > self.max_layer_len or split_point < 1:
            raise ValueError(f"分割点超出范围！请设置在1-{self.max_layer_len}之间。")
        self.num_class = num_classes
        self.split_point = split_point
        self.conv, self.fc = build_model_by_split_point(num_classes, split_point)
        # print(f"self.conv: {self.conv}")
        # print(f"self.fc: {self.fc}")

        # # 预计算总层数（conv层数量 + fc层数量）
        self.total_conv_layers = len(self.get_conv_layers_list())  # 5层（0-4）
        self.total_fc_layers = 0
        if self.fc is not None:
            self.total_fc_layers = len(self.get_fc_layers_list())      # 3层（0-2）

            # self.total_conv_layers = len(self.get_conv_layers_list())  # 11层（0-10）
            # self.total_fc_layers = len(self.get_fc_layers_list())      # 3层（0-2）

        # print(f"模型初始化：总卷积层数={self.total_conv_layers}, 总全连接层数={self.total_fc_layers}")

    def forward(self, x, split_point = None, lower = True, return_features=False):
        """完整模型的前向执行策略（仅负责完整推理）"""
        if split_point is None:
            split_point = self.split_point
        if split_point == training_config.ALEXNET_MAX_SPLIT_POINT:
            # 卷积层执行
            out = self.conv(x)
            # 特征展平
            features = out.view(-1, 256 * 3 * 3)
            # 全连接层执行
            out = self.fc(features)
        else:
            if lower == True:
                if return_features:
                    out, features = self.lower_forward(x=x, split_point=split_point, return_features=return_features)
                else:
                    out = self.lower_forward(x=x, split_point=split_point, return_features=return_features)
            else: # 执行upper逻辑
                if return_features:
                    out, features = self.upper_forward(x=x, split_point=split_point, return_features=return_features)
                else:
                    out = self.upper_forward(x=x, split_point=split_point, return_features=return_features)
        if return_features:
                return out, features
        return out
            
    def lower_forward(self, x, split_point=training_config.ALEXNET_MAX_SPLIT_POINT, return_features=False):
        out = x
        # 1. 先执行卷积层（若分割点在卷积层范围内）
        if split_point < self.max_conv_layers:
            conv_layers = self.get_conv_layers_list()
            for idx in range(split_point):
                out = conv_layers[idx](out)
        # 2. 若分割点跨卷积/全连接层，需先展平特征再执行部分fc层
        elif split_point < self.max_conv_layers + self.max_fc_layers:
            # 先执行完整卷积层 + 展平
            out = self.conv(out)
            features = out.view(-1, 256 * 3 * 3)
            out = features
            # 执行fc层的前 N 层（N = split_point - 总卷积层数）
            fc_start_idx = split_point - self.max_conv_layers
            fc_layers = self.get_fc_layers_list()
            for idx in range(fc_start_idx):
                out = fc_layers[idx](out)
        # 3. 分割点超出总层数（返回完整conv层输出）
        else:
            out = self.conv(out)
            out = out.view(-1, 256 * 3 * 3)
            out = self.fc(out)
        if return_features:
            return out, features
        return out

    def upper_forward(self, x, split_point=training_config.ALEXNET_MAX_SPLIT_POINT, return_features=False):
        out = x
        # 1. 分割点在卷积层范围内：先执行到split_point前的层，再执行剩余层
        if split_point < self.max_conv_layers:
            conv_layers = self.get_conv_layers_list()
            # 执行split_point到最后的卷积层
            for idx in range(split_point, self.max_conv_layers):
                out = conv_layers[idx](out)
            # 展平特征 + 执行完整fc层
            features = out.view(-1, 256 * 3 * 3)
            out = features
            out = self.fc(out)
        # 2. 分割点在fc层范围内：执行split_point后的fc层
        elif split_point < self.max_conv_layers + self.max_fc_layers:
            # 执行fc层从split_point开始的剩余层
            fc_start_idx = split_point - self.max_conv_layers
            fc_layers = self.get_fc_layers_list()
            for idx in range(fc_start_idx, self.max_fc_layers):
                out = fc_layers[idx](out)
        # 3. 分割点超出总层数：执行完整fc层
        else:
            out = self.conv(out)
            features = out.view(-1, 256 * 3 * 3)
            out = features
            out = self.fc(out)
        if return_features:
            return out, features
        return out
    def get_conv_layers_list(self):
        """辅助方法：返回卷积层的可索引列表（供子类分割使用）"""
        return list(self.conv.children())
    
    def get_fc_layers_list(self):
        """辅助方法：返回全连接层的可索引列表（供子类分割使用）"""
        return list(self.fc.children())

if __name__ == "__main__":
    # 测试分割模型的一致性
    full_model = AlexNet(split_point=training_config.ALEXNET_MAX_SPLIT_POINT)

    # 随机输入张量
    input_tensor = torch.randn(1, 3, 32, 32)

    # 完整模型输出
    mid_output = full_model(input_tensor, split_point =1, lower = True)
    print("完整模型中间输出:", mid_output.shape)
    full_output = full_model(mid_output, split_point =1, lower = False)
    print("完整模型输出:", full_output.shape)