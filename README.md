# 介绍
使用两个简单的算法对FashionMNIST进行训练和测试。第一个算法是一个四层的全连接网络，第二个算法是在LeNet-5的基础上增加了两个卷积层，
我称为LeNet-7。全连接网络的代码在model_bp文件中，LeNet-7网络在model_cnn中。
# FashionMNIST数据集
FashionMNIST数据集的地址：https://github.com/zalandoresearch/fashion-mnist。
Fashion MNIST数据集是德国研究机构Zalando Research于2017年8月份，在Github上推出的一个经典数据集。
其中训练集包含60000个样例，测试集包含10000个样例，分为10类，每一类的样本训练样本数量和测试样本数量相同，每个样本都是28×28的灰度图像，
共有10类标签，并且每个样本都有各自唯一的标签。

# 训练
训练过程的epoch设为30，batch_size设为100，训练结束后会将测试的top1准确率、top2准确率和loss曲线画出来。
## 训练全连接网络
在model_bp目录下
```python
python train_bp.py
```
## 训练LeNet-7
在model_cnn目录下
```python
python train_cnn.py
```

# 测试
测试过程会从测试样本中随机取出batch_size个样本进行测试，并将预测的label打印出来。
## 测试全连接网络
在model_bp目录下
```python
python test_bp.py
```
## 测试LeNet-7
在model_cnn目录下
```python
python test_cnn.py
```