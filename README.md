# Huawei-AI-Test
    Huawei ai 比赛 特征提取网络训练
# 说明
    运行src中的main.py 即可开始训练,参数设定在main.py中
    在Network中可以定义网络
    DataLoader负责从aifood_en中读取图片，基于pytorch 的dataloader，可以自己托管数据并生成训练数据
    Trainer负责训练
    从Network中加载网络，从DataLoader中加载数据，初始化Trainer即可训练
 
 # 注意
    为了上传方便，aifood_en中是没有包含数据的，要把images文件夹拷进去
    ！！并且！！在训练时我们只用到了75个类，所以那25个小类的文件夹要挑出来
    可以在Network中替换不同的网络种类，默认为densenet，可以换成resnet，vgg等等
    ！！注意！！，不同网络的输入格式是不一样的,在DataLoder.py中有个transform，
    修改它以让输入可以满足网络输入格式的要求
