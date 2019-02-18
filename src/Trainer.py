"""
Create Trainer to control the process of training
训练器
"""
import numpy as np
from torch import Tensor

class Trainer(object):

    def __init__(self, model,optimizer,lostFunc,dataLoder,para):
        self.model = model
        self.opt = optimizer
        self.crit = lostFunc
        self.dataLoader = dataLoder
        self.epochs = para['num_epoch']
        self.useGPU = para['useGPU']
        self.para = para


    # 训练一批
    def train_single_batch(self, image,labels,batchId):

        if self.useGPU == True:
            image = image.cuda()
            labels = labels.cuda()

        self.opt.zero_grad()
        labels_pred = self.model(image)
        loss = self.crit(labels_pred, labels)
        loss.backward()
        self.opt.step()
        loss = loss.data.cpu().numpy()
        return loss

    # 训练一代
    def train_an_epoch(self, epoch_id):
        assert hasattr(self, 'model'), '没有网络传入'
        self.model.train()

        for ID,smpl in enumerate(self.dataLoader):
            loss = self.train_single_batch(smpl[0],smpl[1],ID)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, ID, loss))


    def train(self):

        for i in range(0,self.epochs):
            self.train_an_epoch(i)
        print('训练完成!')

    # 暂时还不需要测试功能
    def test(self):
        pass
