'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
Usgae:
python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER
'''
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms

from model import PhysicalNN
from model_fast import PhysicalNN_fast
from uwcc import uwcc
import shutil
import os
from torch.utils.data import DataLoader
import sys
from tensorboardX import SummaryWriter

######################### experiment configurations ######

# batchsize = 1
# lr = 0.001
# epochs = 300
# net = PhysicalNN()
# criterion = nn.MSELoss()
# train_aug = False
# pre_trained = None
# tag = 'v0'#作者原版,只训300轮


# batchsize = 1
# lr = 0.001
# epochs = 300
# net = PhysicalNN_fast()
# criterion = nn.MSELoss()
# train_aug = False
# pre_trained = None
# tag = 'v1'#fast版,只训300轮

batchsize = 1
lr = 0.001
epochs = 300
net = PhysicalNN_fast()
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
train_aug = True
# pre_trained = 'results/v1/v1_best.pth'
pre_trained = None
tag = 'v2'#增加随机翻转增强，其他同v1
# tag = 'v2'#以v1_best作为初始化权重，裁剪为小图并数据增强，修改学习率、批次大小、轮数，采用L1损失，不保持v1的优化器状态重头训练

##########################################


save_dir = './results/' + tag

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main():

    best_loss = 9999.0

    n_workers = 0
    ori_fd = '/home/SENSETIME/sunxin/3_datasets/UIEBD/raw-890'
    ucc_fd = '/home/SENSETIME/sunxin/3_datasets/UIEBD/reference-890'
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    #create model
    model = nn.DataParallel(net)

    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if pre_trained is not None:
        checkpoint = torch.load(pre_trained, map_location = torch.device('cuda'))
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    #load data
    trainset = uwcc(ori_dirs, ucc_dirs, train=True, train_aug=train_aug)
    trainloader = DataLoader(trainset, batchsize, shuffle=True, num_workers=n_workers)

    writer = SummaryWriter(save_dir)

    #train
    for epoch in range(epochs):

        tloss = train(trainloader, model, optimizer, criterion, epoch)

        writer.add_scalar('loss', tloss, epoch)

        print('Epoch:[{}/{}] Loss{}'.format(epoch,epochs, tloss))
        is_best = tloss < best_loss
        best_loss = min(tloss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    print('Best Loss: ', best_loss)

def train(trainloader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    model.train()

    for i, sample in enumerate(trainloader):
        ori, ucc = sample
        # print(ori.device, ucc.device)
        ori = ori.cuda()
        ucc = ucc.cuda()

        corrected = model(ori)
        loss = criterion(corrected,ucc)
        losses.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg


def save_checkpoint(state, is_best):
    """Saves checkpoint to disk"""
    freq = epochs * 3 // 10
    epoch = state['epoch'] 

    if epoch%freq==0 and epoch > 0:
        torch.save(state, save_dir + '/' + tag + '_ep_' + str(epoch) + '.pth')

    if is_best:
        torch.save(state, save_dir + '/' + tag + '_best.pth')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
