"""
从网络中提取特征的代码
TODO!!
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob

data_dir = './test'  # train
features_dir = './Resnet_features_test'  # Resnet_features_train

# 从resnet 提取的代码样例
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet50(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output


model = net()
model = model.cuda()

# 四通道图片转换为三通道
def change_image_channels(image):

    if image.mode == 'RGBA':
        r,g,b,a = image.split()
        image = Image.merge('RGB',(r,g,b))
        return image



def extractor(pilImg, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = transform(pilImg)
    print(img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    print(x.shape)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y
