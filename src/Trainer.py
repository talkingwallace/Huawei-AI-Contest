"""
Create Trainer to control the process of training
训练器
"""
import numpy as np
from torch import Tensor
import torch

class Trainer(object):

    def __init__(self, model,optimizer,lostFunc,dataLoder,testLoader,para):
        self.model = model
        self.opt = optimizer
        self.crit = lostFunc
        self.dataLoader = dataLoder
        self.testLoader = testLoader
        self.epochs = para['num_epoch']
        self.useGPU = para['useGPU']
        self.para = para
        self.saveEveryEpochs = para['saveEveryEpochs']
        self.lossLogger = {}


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
            self.lossLogger[ID] = loss


    def train(self):

        for i in range(0,self.epochs):
            self.train_an_epoch(i)
            if (i+1)%self.saveEveryEpochs == 0:
                torch.save(self.model,'epoch_'+str(i)+'_model.pkl')
                print('model of epoch'+str(i)+' saved')
        print('训练完成!')

    # 测试网络
    def test(self):

        totalLoss = 0
        for ID,testData in enumerate(self.testLoader):
            print('testing batch:'+str(ID))
            image = testData[0]
            labels = testData[1]

            if self.useGPU == True:
                image = image.cuda()
                labels = labels.cuda()

            labels_pred = self.model(image.cuda())
            loss = self.crit(labels_pred, labels)
            loss = loss.data.cpu().numpy()
            totalLoss += loss

        print('total loss of '+str(len(self.testLoader))+' batches')
        print(totalLoss)
        print('avg loss:')
        print(totalLoss/len(self.testLoader))

        return totalLoss