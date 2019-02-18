"""
This module is for loading data and data management
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

metaPath = r'../aifood_en/meta/'
imgPath = r'../aifood_en/images/'

# 全局变量

def getSmallSmplFileName():
    """
    获取小数据集名字
    :return:
    """
    f = open(metaPath+r'small_labels_25c.txt','r')
    return f.read().split('\n')

def getBigSmpFileName():
    """
    获取大数据集名字
    :return:
    """
    f = open(metaPath+r'large_labels_75c.txt','r')
    return f.read().split('\n')

# 转换器 把图片转换为可以输入网络的格式
transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()]
)

# 转换函数
# def transformFunc(img):
#     # img 为pillow格式
#     im = transform(img)
#     x = Variable(torch.unsqueeze(im, dim=0).float(), requires_grad=False)
#     return x

# 这个没加转换函数，可以直接看图片
imgBrowser = datasets.ImageFolder(root=imgPath,transform=None)

# 这个加了转换函数，用于生成训练集
foodData = datasets.ImageFolder(root=imgPath,transform=transform)

def getFoodDataLoader(batch_size):
    return torch.utils.data.DataLoader(foodData,batch_size=batch_size,shuffle=True)

