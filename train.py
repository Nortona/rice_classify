import os
import argparse
from sympy import im
from net import resnet18
from nets.vgg16 import Vgg16
from nets.lenet import LeNet
import torch
import torch.nn as nn
import torchvision
import tensorboardX
from utils.callbacks import LossHistory
from utils.utils_fit import fit_one_epoch
from nets.model_training import weights_init
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# from read_rice import train_loader,valid_loader
from read_rice2 import train_loader,valid_loader
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 30
    lr = 0.0001
    batch_size = 32

    step_size = 5  # 每n次epoch更新一次学习率

    save_period = 5   # 多少个epoch保存一次权值

    momentum = 0.937   # 动量因子

    input_shape = [224,224]

    fp16 = True

    weight_decay = 5e-4  # weight_decay(float):权重衰减,

    # net = resnet18().to(device)
    pretrained = False
    model = Vgg16(5,pretrained).to(device)
    if not  pretrained:
        weights_init(model)
    # model = LeNet(5).to(device)
    # print(next(net.parameters()).device)
    # print(net)
    
    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay=weight_decay)
    '''
    params(iterable)-待优化参数的iterable或者定义了参数组的dict
    lr(float):学习率

    momentum(float)-动量因子

    weight_decay(float):权重衰减,使用的目的是防止过拟合.在损失函数中,weight decay是放在正则项前面的一个系数,正则项一般指示模型的复杂度
    所以weight decay的作用是调节模型复杂度对损失函数的影响,若weight decay很大,则复杂的模型损失函数的值也就大.

    dampening:动量的有抑制因子


    optimizer.param_group:是长度为2的list,其中的元素是两个字典
    optimzer.param_group:长度为6的字典,包括['amsgrad','params','lr','weight_decay',eps']
    optimzer.param_group:表示优化器状态的一个字典

    '''


    schedule = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)
    '''
    scheduler 就是为了调整学习率设置的，gamma衰减率为0.5，step_size为10，也就是每10个epoch将学习率衰减至原来的0.5倍。

    optimizer(Optimizer):要更改学习率的优化器
    milestones(list):递增的list,存放要更新的lr的epoch
    gamma:(float):更新lr的乘法因子
    last_epoch:：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。
    默认为-1表示从头开始训练，即从epoch=1
    '''

    # if not os.path.exists("log/vgg16_ep10_lr0.001_bs16"):
    #     os.mkdir("log/vgg16_ep10_lr0.001_bs16")
    # writer = tensorboardX.SummaryWriter("log/vgg16_ep10_lr0.001_bs16")
    save_dir = 'logs'

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()

    model_train = model.train()

    eval_callback = None

    num_train = len(train_loader)
    num_val = len(valid_loader)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")


    # best_acc = 0.0
    for epoch in range(epoch_num):

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        gen = train_loader
        gen_val = valid_loader

        fit_one_epoch(model_train,model,loss_func,loss_history,optimizer,epoch,epoch_step,epoch_step_val,
                      gen,gen_val,epoch_num,device,fp16,scaler,save_period,save_dir)

    loss_history.writer.close()
