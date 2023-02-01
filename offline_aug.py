from pathlib import Path
import cv2
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm

def aug():
    ref_dir = '/home/SENSETIME/sunxin/3_datasets/UIEBD/reference-890'

    save_dir1 = '/home/SENSETIME/sunxin/3_datasets/UIEBD/crop_aug_ref'
    save_dir2 = '/home/SENSETIME/sunxin/3_datasets/UIEBD/crop_aug_raw'

    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)

    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    data_transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(1024,scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    for i, ref_path in enumerate(Path(ref_dir).glob('*.png')):
        
        # if ref_path.name != '6_img_.png': continue

        ref_path = str(ref_path)
        raw_path = ref_path.replace('reference', 'raw')

        ref = Image.open(ref_path)
        raw = Image.open(raw_path)

        j = 0
        for seed in range(20):

            # try:
            torch.random.manual_seed(seed)
            ref = data_transform_aug(ref)
            
            tmp = ref.convert('L') 
            tmp = np.asarray(tmp)
            gx,gy = np.gradient(tmp)
            g = np.mean(abs(gx + gy))

            # if g < 1.0: continue

            ref.save(save_dir1 + '/' + str(i) + '_' + str(j) + '.png')

            torch.random.manual_seed(seed)
            raw = data_transform_aug(raw)
            raw.save(save_dir2 + '/' + str(i) + '_' + str(j) + '.png')

            print(save_dir1 + '/' + str(i) + '_' + str(j) + '.png')
            j += 1
            # except:
            #     continue

def down():
    ref_dir = '/home/SENSETIME/sunxin/3_datasets/UIEBD/crop_aug_ref'

    save_dir1 = '/home/SENSETIME/sunxin/3_datasets/UIEBD/crop_aug_ref_256x256'
    save_dir2 = '/home/SENSETIME/sunxin/3_datasets/UIEBD/crop_aug_raw_256x256'

    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)

    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    for i, ref_path in enumerate(Path(ref_dir).glob('*.png')):
        
        ref_path = str(ref_path)
        raw_path = ref_path.replace('ref', 'raw')

        ref = Image.open(ref_path)
        raw = Image.open(raw_path)

        ref = ref.resize((256, 256))
        raw = raw.resize((256, 256))

        ref.save(ref_path.replace('crop_aug_ref', 'crop_aug_ref_256x256'))
        raw.save(raw_path.replace('crop_aug_raw', 'crop_aug_raw_256x256'))

if __name__=='__main__':
    aug()
    down()
