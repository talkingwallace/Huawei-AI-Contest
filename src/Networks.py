"""
设（复）计（制）的网络
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms
from torch.nn import Linear

"""
使用densenet进行分类训练
"""
class DenseNet(nn.Module):

    # densnet的模型由pytorch热情提供
    # 可以从pytorch上下载各种网络结构

    def __init__(self,categoriesNum):
        super(DenseNet,self).__init__()
        self.categoriesNum = categoriesNum
        self.densenet = models.densenet121(pretrained=False)
        self.densenet.classifier = Linear(in_features=1024,out_features=categoriesNum,bias=True)

    def forward(self, image):
        output = self.densenet(image)
        return output

