# https://blog.csdn.net/qq_34611579/article/details/84330968 参考该文

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms,datasets
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sb # 画图库
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PIL import Image
from torch.utils.data import Dataset

imgPath = r'../aifood_en/smallSmplData'

# 一些处理的参数 最后提取特征向量，要做多一次池化
para = {
    'kernel':7, # 7 by 7 的一个filter
    'stride':1, # 步长
}

# load a net work here


# class resnet(nn.Module):
#     def __init__(self):
#         super(resnet, self).__init__()
#         self.net = models.resnet50(pretrained=True) # 尝试用pretrain resnet提取
#
#     def forward(self, input):
#         output = self.net.conv1(input)
#         output = self.net.bn1(output)
#         output = self.net.relu(output)
#         output = self.net.maxpool(output)
#         output = self.net.layer1(output)
#         output = self.net.layer2(output)
#         output = self.net.layer3(output)
#         output = self.net.layer4(output)
#         output = self.net.avgpool(output)
#         return output

model = torch.load('final_model_dense.pkl') # 训练过的densenet
model = model.densenet

# model = models.densenet121(pretrained=True).cuda() # pretrain 的densenet提取

# model = resnet().cuda() # pretrain 的 resnet

# model = torch.load('final_model_res.pkl').cuda() # 这个读不进来

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()]
)

smallSmpls = datasets.ImageFolder(root=imgPath,transform=transform) # 读取数据

def extractor_dense(imgTensor,model): # densenet 用这个
    # 生成一个 1024维度的特征向量
    # avg 池化
    avg = nn.AvgPool2d(para['kernel'], para['stride'])
    x = Variable(torch.unsqueeze(imgTensor, dim=0).float(), requires_grad=False)
    return avg(model.features(x.cuda())).flatten().cpu().detach().numpy() # 可真麻烦 绕死我了

def extractor_resnet(imgTensor,model): # resnet 用这个

    x = Variable(torch.unsqueeze(imgTensor, dim=0).float(), requires_grad=False)
    y = model(x.cuda()).cpu().detach().numpy()
    return y.flatten()

# 设定extractor
extractor = extractor_resnet


# 提取向量 准备画图
print('extracting vecs')
vecList = []
classList = []

for i in smallSmpls:
    vec = extractor(i[0],model)
    vecList.append(vec)
    classList.append(i[1])

vecList = np.array(vecList)

# 把数字标签转为字符串方便可视化
strLabel = []
for i in classList:
    strLabel.append(smallSmpls.classes[i])

# 对 1024 维度的向量使用 tsne / PCA 降维
print('reducing dimensions')
twoDvec = TSNE(n_components=2).fit_transform(vecList)
# twoDvec = PCA(n_components=2).fit_transform(vecList) # 使用PCA x要乘以 10^6 y要乘以10^7 , 不然x,y太小了
x = twoDvec[:,0]
y = twoDvec[:,1]
df = pd.DataFrame()
df['x'] = x
df['y'] = y
df['class'] = strLabel
df = df[0:100]
ax = sb.scatterplot(x="x", y="y", hue="class", data=df)
plt.show()
