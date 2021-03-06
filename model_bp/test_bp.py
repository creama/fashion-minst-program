import torch
import torchvision.transforms as tranforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import model_bp
import visualize

# 处理数据集
data_dir = '../data/'
tranform = tranforms.Compose([tranforms.ToTensor()])

test_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.figure()
visualize.imshow_batch(next(iter(test_dataloader)))

# 加载模型
model_bp = model_bp.Classifier()
model_bp.load_state_dict(torch.load(f='model/model_bp_final_epoch.pth', map_location=device))
print(model_bp)

# 测试
print('\ntest start:')
images, labels = next(iter(test_dataloader))
outputs = model_bp(images)
_, prediction = torch.max(outputs, 1)
print('label:', labels)
print('prdeiction:', prediction)

plt.show()
