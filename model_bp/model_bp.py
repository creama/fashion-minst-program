from torch import nn
import torch.nn.functional as f


class Classifier(nn.Module):
    """
    将输入的二维图片张开为一维张量，使用4层的全连接网络
    输入：长度为784的一阶张量
    输出：长度为10的一阶张量
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    # 前向传播
    def forward(self, x):
        # 确保输入的tensor是展开的单列数据，把每张图片的通道、长度、宽度三个维度都压缩为一列
        x = x.view(x.shape[0], -1)

        # 使用ReLU和Dropout
        out = self.dropout(f.relu(self.fc1(x)))
        out = self.dropout(f.relu(self.fc2(out)))
        out = self.dropout(f.relu(self.fc3(out)))

        # 在输出单元不需要使用Dropout方法，使用LogSoftmax
        out = f.log_softmax(self.fc4(out), dim=1)

        return out
