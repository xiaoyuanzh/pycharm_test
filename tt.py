import os
import timeit
import torch                     # pytorch 最基本模块
import torch.nn as nn            # pytorch中最重要的模块，封装了神经网络相关的函数
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.optim as optim      # 优化模块，封装了求解模型的一些优化器，如Adam SGD
from torch.optim import lr_scheduler # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  #pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  #pytorch 视觉库提供了加载数据集的接口

from tqdm import tqdm

DATA_DIR = os.path.join(os.getcwd(), "data")

BATCH_SIZE = 64

EPOCHES = 3

# 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.1

# 加载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,)) # 数据规范化到正态分布
                    ])),
    batch_size=BATCH_SIZE, shuffle=True) # 指明批量大小，打乱，这是处于后续训练的需要。
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        # 提取特征层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            #最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 分类层
        self.classifier = nn.Sequential(
            # Dropout层
            # p = 0.5 每个权重有0.5可能性为 0
            nn.Dropout(p=0.5),
            nn.Linear(64*7*7,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,10)

        )

    def forward(self, x):
        x = self.features(x)
        # 输出结果成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  x


# 初始化模型
ConvModel = ConvNet()
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(DEVICE)
# 定义模型优化器
optimizer = torch.optim.Adam(ConvModel.parameters(), lr = learning_rate)
# 定义学习率调度器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)


def train(num_epochs,_model, _device, _train_loader, _optimizer, _lr_scheduler):
    # pass
    _model.train()
    _lr_scheduler.step()
    for epoch in range(num_epochs):
        # print(epoch)
        start = end = 0
        for i, (images,labels) in enumerate(_train_loader):
            if (i+1) % 100 == 1:
                start = timeit.default_timer()
            samples = images.to(_device)
            labels = labels.to(_device)

            output = _model(samples.reshape(-1,1,28,28))
            loss = criterion(output, labels)
            # print('loss:{}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                end = timeit.default_timer()
                print("Epoch:{}/{}, Time:{}s, step:{}, loss:{:.4f}".format(epoch + 1, num_epochs, end - start, i + 1,
                                                                           loss.item()))




for epoch in tqdm(range(1, EPOCHES+1)):
    train(epoch, ConvModel, DEVICE, train_loader, optimizer, exp_lr_scheduler)


