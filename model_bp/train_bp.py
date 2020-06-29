import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as tranforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import model_bp


# 设置barch_size和epoch
batch_size = 100
epoch_num = 30

# 处理数据集
data_dir = '../data/'
tranform = tranforms.Compose([tranforms.ToTensor()])

# 下载FashionMNIST训练集数据，如果已经下载了就不会自动下载
train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, download=True, train=True, transform=tranform)
val_dataset = torchvision.datasets.FashionMNIST(root=data_dir, download=True, train=False, transform=tranform)

# 构建数据载入器,每次从数据集中载入batch_size张图片，每次载入都打乱顺序
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


# 定义网络
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 实例化
model_bp = model_bp.Classifier()
print(model_bp)
model_bp = model_bp.to(device)

# 交叉熵损失
loss_fc = nn.CrossEntropyLoss()
# 随机梯度下降优化，初始学习率为0.001
optimizer = optim.SGD(model_bp.parameters(), lr=0.001, momentum=0.9)
# 每20个epoch更新学习率
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# 训练
# 保存训练损失和测试准确率，最后画曲线图
file_avg_loss = open('log/avg_loss.txt', 'w')
file_test_accuarcy01 = open('log/test_accuracy_top1.txt', 'w')
file_test_accuarcy02 = open('log/test_accuracy_top2.txt', 'w')

print('\ntrain start:epoch={}, batch_size={}, device={}'.format(epoch_num, batch_size, device))
for epoch in range(epoch_num):
    # 每10个epoch，保存模型
    if epoch >=10 and epoch % 10 == 0:
        torch.save(model_bp.state_dict(), 'model/model_bp_' + str(epoch) + '_epoch.pth')
    sum_loss = 0.0
    accuracy01 = 0.0
    accuracy02 = 0.0
    scheduler.step()
    for i, sample_batch in enumerate(train_dataloader):

        inputs = sample_batch[0]
        labels = sample_batch[1]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # 打开训练模式
        model_bp.train()
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()
        outputs = model_bp(inputs)
        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        print('iter:', i, '\t', 'loss:', loss.item())

        sum_loss += loss.item()
        # 每batch_size次iter，统计一次数据
        if i >= batch_size-1 and i % (batch_size-1) == 0:
            correct01 = 0
            correct02 = 0
            total = 0

            # 开始计时
            time_start = time.clock()

            # 关闭Dropout
            model_bp.eval()
            for inputs, labels in val_dataloader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model_bp(inputs)

                # 计算topk
                topk_p, topk_class = outputs.topk(2, 1)
                topk_class = topk_class.cpu().numpy()
                labels = labels.cpu().numpy()
                correct01 += (topk_class[:, 0] == labels).sum().item()
                correct02 += (((topk_class[:, 0] == labels) + (topk_class[:, 1] == labels)).sum()).item()
                total += batch_size

            # 测试10000张图片的fps
            fps = 10000/(time.clock() - time_start)
            # 得到10000张测试图片的topk准确率
            accuracy01 = correct01 / total
            accuracy02 = correct02 / total
            print('[iter/epoch]=[{}/{}] avg_loss = {:.5f} accuracy_top1 = {:.5f} accuracy_top2 = {:.5f}'.format(
                                                        i+1, epoch + 1, sum_loss / batch_size, accuracy01, accuracy02))
            file_avg_loss.write(str(sum_loss / batch_size) + '\n')
            file_test_accuarcy01.write(str(accuracy01)+'\n')
            file_test_accuarcy02.write(str(accuracy02)+'\n')
            sum_loss = 0.0

# 训练结束
print('\n train finished')
print(' fps=', fps)
torch.save(model_bp.state_dict(), 'model/model_bp_final_epoch.pth')
file_test_accuarcy01.close()
file_test_accuarcy02.close()
file_avg_loss.close()

# 画损失和准确率的曲线图
testing_accuracy01 = []
testing_accuracy02 = []
training_loss = []
file_avg_loss = open('log/avg_loss.txt', 'r')
file_test_accuarcy01 = open('log/test_accuracy_top1.txt', 'r')
file_test_accuarcy02 = open('log/test_accuracy_top2.txt', 'r')

for eachline in file_test_accuarcy01:
    testing_accuracy01.append(eachline)

for eachline in file_test_accuarcy02:
    testing_accuracy02.append(eachline)

for eachline in file_avg_loss:
    training_loss.append(eachline)

plt.plot(testing_accuracy01, label='testing_accuracy_top1')
plt.plot(testing_accuracy02, label='testing_accuracy_top2')
plt.plot(training_loss, label='training_loss')
plt.legend()
plt.show()

file_test_accuarcy01.close()
file_test_accuarcy02.close()
file_avg_loss.close()
