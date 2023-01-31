'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
Usgae:
python test.py --checkpoint CHECKPOINTS_PATH
'''
import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
from model_fast import PhysicalNN_fast
from torchvision import transforms
import time
from evaluate import rmetrics, nmetrics, print_np_info
from ptflops import get_model_complexity_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
testtransform = transforms.Compose([transforms.ToTensor(),])
unloader = transforms.ToPILImage()

def run_test(net, checkpoint_path, test_img_dir, do_metric):
    model_name = checkpoint_path.split('/')[-1].split('.pth')[0]

    model = torch.nn.DataParallel(net).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model.eval()
    model.to(device)

    FLOPs1080p, Parms1080p = get_model_complexity_info(model, (3, 1080, 1920), as_strings=True, print_per_layer_stat=False, verbose=False)

    UIQM, UCIQE, PSNR, SSIM, FPS, cnt = 0, 0, 0, 0, 0, 0
    for img_name in os.listdir(test_img_dir):

        if 'of' in img_name: 
            continue
        
        start_time = time.time()

        img = Image.open(test_img_dir + '/' + img_name)
        inp = testtransform(img).unsqueeze(0)
        with torch.no_grad():
            inp = inp.to(device)
            out = net(inp)
        corrected = unloader(out.cpu().squeeze(0))

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        FPS += fps

        save_path = test_img_dir + '/{}_of_{}.png'.format(img_name, model_name)
        corrected.save(save_path)

        if do_metric:
            corrected = Image.open(save_path)
            corrected = np.asarray(corrected)

            uiqm,uciqe = nmetrics(corrected)
            UIQM += uiqm
            UCIQE += uciqe

            if os.path.exists(test_img_dir + '.jpg'):
                reference = Image.open(test_img_dir + '.jpg')
                reference = np.asarray(reference)
                psnr,ssim = rmetrics(corrected,reference)
            else:
                psnr,ssim = 0, 0
            PSNR += psnr
            SSIM += ssim

        cnt += 1

    FPS = FPS / cnt
    UIQM = UIQM / cnt
    UCIQE = UCIQE / cnt
    PSNR = PSNR / cnt
    SSIM = SSIM / cnt

    out_info = 'pth:{}, FLOPs1080p:{}, Parms1080p:{}, data:{}, FPS:{}, UIQM:{}, UCIQE:{}, PSNR:{}, SSIM:{}'.format(checkpoint_path, FLOPs1080p, Parms1080p, test_img_dir, FPS, UIQM, UCIQE, PSNR, SSIM)

    return out_info

def main():
    do_metric = False

    data_dirs = [
        './test_img/ori', 
        './test_img/haze',
        # './test_img/turbid/A',
        # './test_img/turbid/B',
        # './test_img/turbid/C',
        # './test_img/turbid/D',
    ]

    models = [
        # (PhysicalNN(), 'results/ori/ori_ep_2842.pth'),
        # (PhysicalNN(), 'results/v0/v0_best.pth'),
        # (PhysicalNN_fast(), 'results/v1/v1_best.pth'),
        (PhysicalNN_fast(), 'results/v2/v2_best.pth'),
    ]

    all_info = []
    for test_img_dir in data_dirs:
        for net, checkpoint_path in models:
            out_info = run_test(net, checkpoint_path, test_img_dir, do_metric)
            print(out_info)
            all_info.append(out_info)

    with open('results/compare.txt', 'w') as f:
        for info in all_info:
            f.write(info)
            f.write('\n')

if __name__ == '__main__':
    main()