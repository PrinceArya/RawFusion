"""
We refer the code made from  
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

import numpy as np
import argparse

import os, sys, time, shutil

from PIL import Image
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()
from torchmetrics.image import PeakSignalNoiseRatio

from DataLoader.custom_data_class import CustomDataset
from models.Model_02_MFP import Merging_Net as My_model
import pdb

from utils.utils import *
from utils.checkpoint import *


def eval(model, cuda=True, mGPU=True):
    print('Eval Process......')

    # dataset and dataloader
    data_set = CustomDataset(root_dir="../../datasets/val/", transform=transforms.ToTensor(), train=False)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0

        for i, (burst_noise, gt) in enumerate(data_loader):

            if cuda:
                burst_noise = burst_noise.cuda()
                gt = gt.cuda()

            burst_noise = burst_noise.squeeze(2)
            pred = model(burst_noise)
            
            psnr_t = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim_t = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            # ssim_t=psnr_t
            psnr += psnr_t
            ssim += ssim_t
            pred = torch.clamp(pred, 0.0, 1.0)
    #print('Average PSNR on Val: {:.2f}dB'.format(psnr/(i+1)))
    return psnr/(i+1)

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()

        self.sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype = torch.float32)
        self.sobel_y = torch.tensor([[[[1, 2,  1], [0, 0, 0], [-1, -2, -1]]]], dtype = torch.float32)

    def forward(self, pred, target):
        pred_gray = 0.299*pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299*target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(target.device)

        grad_pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)

        grad_target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target_gray, self.sobel_y, padding=1)

        edge_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        edge_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)

        final_loss = F.l1_loss(edge_pred, edge_target)

        return final_loss  

class CharbonnierLoss(nn.Module):    
    def __init__(self, eps=1e-3, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss   

class weighted_MSELoss(nn.Module):    
    def __init__(self):
        super(weighted_MSELoss, self).__init__()

    def forward(self, pred, target):
        
        error = torch.abs(pred-target).detach()
        w = 1+4*error 
        w = torch.clamp(w,1, 5)
        loss = (w*(pred-target)**2).mean()
        return loss                 

class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()

def train(num_threads, cuda, restart_train, mGPU):
    torch.set_num_threads(num_threads)

    batch_size = 4
    lr_decay = 0.99
    lr = 8e-4
    #lr = 2e-5

    n_epoch =10000

    # checkpoint path
    checkpoint_dir = 'checkpoint_dir'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # output path
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # logs path
    logs_dir = 'logs_dir'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    shutil.rmtree(logs_dir)

    # dataset and dataloader

    data_set = CustomDataset(root_dir="../../datasets/trn/", transform=transforms.ToTensor(), train=True)
    #data_set = CustomDataset(root_dir="../datasets/trn/", transform=train_transform, train=True)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    print("Length of the data_loader :", len(data_loader))
    # model here
    # model = UNet(in_channels=9,  # 9 frames considered as channel dimension
    #     n_classes=3,        # out channels (RGB)
    #     depth=4,
    #     wf=6,
    #     padding=True,
    #     batch_norm=False,
    #     up_mode='upconv')
    model = My_model()

    print('\n-------Training started -------\n')

    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)
    model.train()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.004)
    optimizer.zero_grad()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_decay)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=2e-7) 

    average_loss = MovingAverage(200)
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            #best_loss = checkpoint['best_loss']
            best_loss = np.inf
            best_psnr = 0
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            best_psnr = 0
            print('=> no checkpoint file to be loaded.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        best_psnr = 0
        if os.path.exists(checkpoint_dir):
            pass
        else:
            os.mkdir(checkpoint_dir)
        print('=> training')
    
    #charbonnierloss = CharbonnierLoss()
    #l1loss = nn.L1Loss()
    # mssim_loss = MS_SSIM_L1_LOSS() #nn.MSELoss()
    #edge_loss = SobelEdgeLoss()
    psnr_loss = PeakSignalNoiseRatio(data_range=1.0).cuda()
    #mse_loss = nn.MSELoss()

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        print('='*20, 'lr={}'.format([param['lr'] for param in optimizer.param_groups]), '='*20)
        avg_loss = 0
        avg_psnr = 0
        avg_ssim = 0
        avg_step = 0
        for step, (burst_noise, gt) in enumerate(data_loader):
            t0 = time.time()
            if cuda:
                burst_noise = burst_noise.cuda()
                gt = gt.cuda()
            burst_noise = burst_noise.squeeze(2)
            pred = model(burst_noise)
            #loss = mse_loss(pred, gt)
            #loss = 0.64*l1loss(pred, gt) + 0.16*mssim_loss(pred, gt) + 0.20*edge_loss(pred, gt)
            loss = -1*psnr_loss(pred, gt)
            #loss = l1loss(pred,gt) + 0.2*edge_loss(pred,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)
            # pdb.set_trace()

            psnr = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            avg_loss += loss.item()
            avg_psnr += psnr
            avg_ssim += ssim
            avg_step += 1
            t1 = time.time()
            # save images
            if (epoch % 50 == 0) and (step < 20):
                for frame in range(9):
                    pil_image = to_pil_image(burst_noise[0][frame])
                    pil_image.save(f'./{output_dir}/Batch{step}_input{frame}.png')
                pil_image = to_pil_image(gt[0])
                pil_image.save(f'./{output_dir}/Batch{step}_gt.png')
                pil_image = to_pil_image(pred[0])
                pil_image.save(f'./{output_dir}/Batch{step}_output_E{epoch}.png')
            # print
            if step % 5 == 0:
                print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t|'
                      ' loss: {:.4f}\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| time:{:.2f} seconds.'
                      .format(global_step, epoch, step, loss, psnr, ssim, t1-t0))
            global_step += 1
        print('Epoch {} is finished, time elapsed {:.2f} seconds.'.format(epoch, time.time() - epoch_start_time))
        print('Average loss : {:.5f}\t| Average PSNR : {:.3f}\t| Average SSIM : {:.3f} \n'.format(avg_loss/avg_step, avg_psnr/avg_step, avg_ssim/avg_step))
        if epoch % 5 == 0:
            avg_psnr_val = eval(model)
            print('********Average Validation PSNR : {:.3f}dB ****************'.format(avg_psnr_val))
            if avg_psnr_val > best_psnr:
                is_best = True
                best_psnr = avg_psnr_val
            # if average_loss.get_value() < best_loss:
            #     is_best = True
            #     best_loss = average_loss.get_value()
            else:
                is_best = False

            save_dict = {
                'epoch': epoch,
                'global_iter': global_step,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict()
            }
            save_checkpoint(
                save_dict, is_best, checkpoint_dir, global_step, max_keep=5
            )


        # decay the learning rate
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > 1e-7:
            scheduler.step()
        else:
            for param in optimizer.param_groups:
                param['lr'] = 1e-7



if __name__ == '__main__':
    train(num_threads=1, cuda=True, restart_train=False, mGPU=0)

