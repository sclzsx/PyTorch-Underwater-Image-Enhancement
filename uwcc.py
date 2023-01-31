'''
Authot:Xuelei Chen(chenxuelei@hotmail.com)
'''
import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from evaluate import print_np_info
from matplotlib import pyplot as plt

def img_loader(path):
    img = Image.open(path)
    return img

def get_imgs_list(ori_dirs,ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        img_name = (ori_imgdir.split('/')[-1]).split('.')[0]
        ucc_imgdir = os.path.dirname(ucc_dirs[0])+'/'+img_name+'.png'

        if ucc_imgdir in ucc_dirs:
            img_list.append(tuple([ori_imgdir,ucc_imgdir]))

    return img_list

class uwcc(data.Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader, train_aug=False):
        super(uwcc, self).__init__()

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.train = train
        self.loader = loader

        if self.train == True:
            print('Found {} pairs of training images'.format(len(self.img_list)))
        else:
            print('Found {} pairs of testing images'.format(len(self.img_list)))

        self.train_aug = train_aug
            
    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        data_transform_aug = transforms.Compose([
                # transforms.RandomResizedCrop(256,scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),

                transforms.ToTensor(),
            ])

        if self.train == True and self.train_aug == True:
            seed = torch.random.seed()

            torch.random.manual_seed(seed)
            sample[0] = data_transform_aug(sample[0])

            torch.random.manual_seed(seed)
            sample[1] = data_transform_aug(sample[1])
        else:
            sample[0] = data_transform(sample[0])
            sample[1] = data_transform(sample[1])

        return sample

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    ori_fd = '/home/SENSETIME/sunxin/3_datasets/UIEBD/raw-890'
    ucc_fd = '/home/SENSETIME/sunxin/3_datasets/UIEBD/reference-890'
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    trainset = uwcc(ori_dirs, ucc_dirs, train=True, train_aug=True)

    unloader = transforms.ToPILImage()

    for i, sample in enumerate(trainset):
        src, dst = sample[0], sample[1]

        diff = torch.abs(src - dst)

        src = unloader(src)
        dst = unloader(dst)
        diff = unloader(diff)

        plt.subplot(131)
        plt.imshow(src)

        plt.subplot(132)
        plt.imshow(dst)

        plt.subplot(133)
        plt.imshow(diff)

        plt.show()