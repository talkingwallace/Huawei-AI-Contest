import torch
import DataLoader
import Trainer
import Networks


# 控制台测试用
# import os
# os.chdir(r'D:\Huawei AI Test\src')
# import sys
# sys.path.append(r'D:\Huawei AI Test')

para = {
    'num_epoch': 100, # 代数
    'batch_size': 16,
    'lr': 0.01, # learning rate
    'categoryNums':75, # 75个类
    'useGPU':True, # 开启GPU模式
}

# 数据加载
foodDataLoader = DataLoader.getFoodDataLoader(para['batch_size'])

# 网络 75个类，所以参数是75
Net = Networks.DenseNet(para['categoryNums'])
if para['useGPU'] == True:
    Net = Net.cuda()

# 使用SGD优化器 传入网络参数与学习率lr
optimizer = torch.optim.SGD(Net.parameters(),lr=para['lr'])

# 损失函数 交叉熵
lossFunc = torch.nn.CrossEntropyLoss()

# 训练器 传入优化器，网络，损失函数，数据加载器，还有参数
trainer = Trainer.Trainer(Net,optimizer,lossFunc,foodDataLoader,para)

# 开始训练
trainer.train()

# 保存
torch.save(Net,'model.pkl')


