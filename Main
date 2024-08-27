# -*- coding:utf-8 -*-
# author:ranshuhao

import argparse
import logging
import os
import pprint
import random
import shutil
import sys
from copy import deepcopy
import time
import cv2
import numpy as np
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.LRSTTC import LRSTTCDataset
from dataset.Luding import LudingDataset
from dataset.jinshajiang import JinshajiangDataset
from dataset.sichuan import SichuanDataset
from tool import ramps
from dataset.nepal import NepalDataset
from more_scenarios.medical.model.unet import UNet, kaiming_normal_init_weight
from dataset.semi import SemiDataset
from dataset.bijie import BijieDataset
from supervised import evaluate
from tool.Loss import abCE_loss
from tool.early_stopping import EarlyStopping
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.dist_helper import setup_distributed
from tool.lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, \
    transforms_back_scale, postprocess_scale
from more_scenarios.medical.util.utils import AverageMeter, count_params, init_log, DiceLoss
import torch.nn.functional as F
from PIL import Image
import skimage.io as io
from torchmetrics.functional import accuracy

parser = argparse.ArgumentParser(description='UniMeanMatch3')

parser.add_argument('--config', type=str, default=r'configs/bijie.yaml')
parser.add_argument('--labeled-id-path', type=str, default=r'E:\DeepLearning\Code\UniMatch\UniMatch-new\splits\bijie\10\labeled.txt')
parser.add_argument('--unlabeled-id-path', type=str, default=r'E:\DeepLearning\Code\UniMatch\UniMatch-new\splits\bijie\10\unlabeled.txt')
parser.add_argument('--val-id-path', type=str, default=r'E:\DeepLearning\Code\UniMatch\UniMatch-new\splits\bijie\10\val_id.txt')
parser.add_argument('--save-path', type=str, default=r'E:\DeepLearning\Code\UniMatch\UniMatch-new\result_bijie')

# Method options
parser.add_argument('--ema_decay', default=0.999, type=float)
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')  # 一致性损失 k*e的（-5*（1-T）*（1-T）））次方，这里就是k，即能达到的最大权重值
parser.add_argument('--consistency_rampup', type=float, default=5, help='consistency_rampup')  # 上面公式里的T，训练次数增长到这个轮次数后，一致性权重达到最大


class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ConvBlock':
        m.weight.data.normal_(0.0, 0.02)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def create_model(chas, classnum, channs, ema=False):
    if ema:
        model = UNet(in_chns=chas, class_num=classnum, channel=channs, dropout=[0.00, 0.0, 0.0, 0.0, 0.0])
        model = kaiming_normal_init_weight(model)
        model = model.cuda()
        for param in model.parameters():
            param.detach_()  # 使得参数无法梯度更新
    else:
        model = UNet(in_chns=chas, class_num=classnum, channel=channs, dropout=[0.00, 0.0, 0.0, 0.0, 0.0])
        model.apply(weights_init)
        model = model.cuda()
    return model


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)






def compute_iou(prediction, target):
    intersection = torch.logical_and(prediction, target)
    union = torch.logical_or(prediction, target)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    log_savePath = args.save_path + '\log.log'
    logger = init_log('global', log_savePath, logging.INFO)
    logger.propagate = 0
    shutil.copy(sys.argv[0], args.save_path + '\\')
    shutil.copy(sys.path[0] + r'\more_scenarios\medical\model\unet.py', args.save_path)
    for j in range(10):
        start = time.time()
        seed = 3407
        torch.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        np.random.seed(seed)
        random.seed(seed)
        # cuDNN设置
        deterministic = True
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True  
        else:
            torch.backends.cudnn.benchmark = True  
            torch.backends.cudnn.deterministic = False 
        torch.backends.cudnn.enabled = True  

        if cfg['dataset'] == 'Bijie':
            trainset_u = BijieDataset(cfg['dataset'], 'train_u', cfg['crop_size'], args.unlabeled_id_path)
            trainset_l = BijieDataset(cfg['dataset'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
            valset = BijieDataset(cfg['dataset'], 'val', id_path=args.val_id_path)
        else:
            print('数据集选择出错，请重新选择！')
            return

        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=True, persistent_workers=True, shuffle=True)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=True, persistent_workers=True, shuffle=True)
        trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=True, persistent_workers=True, shuffle=True)
        valloader = DataLoader(valset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=False, persistent_workers=True, shuffle=True)

        all_args = {**cfg, **vars(args), 'ngpus': 1}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

        channs = [32, 64, 128, 256, 512]
        model = create_model(chas=cfg['inchannel'], channs=channs, classnum=cfg['nclass'])
        ema_model = create_model(chas=cfg['inchannel'], channs=channs, classnum=cfg['nclass'], ema=True)

        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)  
        criterion_ce = nn.CrossEntropyLoss()
        criterion_dice = DiceLoss(n_classes=cfg['nclass'])
        early_stopping = EarlyStopping(patience=7, verbose=True, path=os.path.join(args.save_path, 'best_' + str(j) + '.pth'), log=logger)

        total_iters = len(trainloader_u) * cfg['epochs']
        epoch = -1

        for epoch in range(epoch + 1, cfg['epochs']):
            logger.info('=====> Epoch: {:}, LR: {:.5f}, Previous best: {}'.format(epoch, optimizer.param_groups[0]['lr'], early_stopping.best_score))
            total_loss = AverageMeter()
            total_loss_x = AverageMeter()
            total_loss_mse = AverageMeter()
            total_loss_s1 = AverageMeter()
            total_loss_s2 = AverageMeter()
            total_loss_w_fp = AverageMeter()
            total_loss_w_noise = AverageMeter()
            total_mask_ratio = AverageMeter()

            loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

            for i, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):

                img_x, mask_x = img_x.cuda(), mask_x.cuda()
                img_u_w = img_u_w.cuda()
                img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
                cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
                img_u_w_mix = img_u_w_mix.cuda()
                img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

                model.train(True)
                ema_model.train(True)

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                preds, pred_u_w_fp_Drop = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])

                if epoch > 10:
                    anchorMin = pred_u_w.softmax(dim=1).argmin(dim=1)
                    cutmix_box1 = torch.mul(cutmix_box1, anchorMin)

            
                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
             
                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)


                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]


                loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(), ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())

                meanteancher_weight = get_current_consistency_weight(epoch, args.consistency, args.consistency_rampup)
                if cfg['dataset'] == 'Nepal':
                    loss_u_mse = criterion_ce(outputs_u_w_ema.softmax(dim=1), mask_u_w) * meanteancher_weight 
                else:
                    loss_u_mse = criterion_dice(outputs_u_w_ema.softmax(dim=1), mask_u_w.unsqueeze(1).float()) * meanteancher_weight 

                loss_u_w_fp_Drop = criterion_dice(pred_u_w_fp_Drop.softmax(dim=1), mask_u_w.unsqueeze(1).float(), ignore=(conf_u_w < cfg['conf_thresh']).float())
                loss = (loss_x + loss_u_s1 * 0.33 + loss_u_w_fp_Drop * 0.33 + loss_u_mse * 0.33) / 2.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iters = epoch * len(trainloader_u) + i
                update_ema_variables(model, ema_model, args.ema_decay, iters)

                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_mse.update(loss_u_mse.item())
                total_loss_s1.update(loss_u_s1.item())
                total_loss_w_fp.update(loss_u_w_fp_Drop.item())

                mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
                total_mask_ratio.update(mask_ratio.item())


                if i % (len(trainloader_u) // 6) == 0:
                    logger.info(
                        'Iters: {:}, Total loss: {:.3f},  监督Loss: {:.3f},  MeanTeacherLoss：{:.3f},  强增强1Loss: {:.3f},  强增强2Loss: {:.3f},  '
                        '强弱特征扰动DLoss: {:.3f}, Mask ratio: ''{:.3f}'
                        .format(i, total_loss.avg, total_loss_x.avg, total_loss_mse.avg, total_loss_s1.avg,
                                total_loss_s2.avg,
                                total_loss_w_fp.avg, total_mask_ratio.avg))

            ema_model.eval()
            total_LOSS = 0.0
            total_IoU = 0.0
            with torch.no_grad():
                for img, mask in valloader:
                    img, mask = img.cuda(), mask.cuda()
                    pred = ema_model(img)
                    if cfg['dataset'] == 'Nepal':
                        total_LOSS += criterion_dice(pred.softmax(dim=1), mask.unsqueeze(1).long())
                    else:
                        total_LOSS += criterion_ce(pred, torch.squeeze(mask).long())
                        total_IoU += compute_iou(pred.argmax(dim=1), mask)
            mean_loss = total_LOSS / len(valloader)
            mean_IoU = total_IoU / len(valloader)
            logger.info('***** Evaluation ***** >>>> Mean_Loss: {:.2f}\n'.format(mean_loss))
            logger.info('***** Evaluation ***** >>>> Mean_IoU: {:.2f}\n'.format(mean_IoU))
            writer.add_scalar('eval/Mean_Loss', mean_loss, epoch)
            writer.add_scalar('eval/Mean_IoU', mean_IoU, epoch)
            early_stopping(mean_IoU, ema_model)
            if early_stopping.early_stop:
                logger.info("val_IoU 超过 {} 轮未升高，最佳IoU为： {} ，此时Epoche数为： {} ，训练结束".format(early_stopping.patience, early_stopping.best_score, epoch))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, 'last_' + str(j) + '.pth'))

        writer.close()
        time_elapsed = time.time() - start
        logger.info('总训练时间为 {:d}m {:.2f}s'.format(int(time_elapsed // 60), time_elapsed % 60))


if __name__ == '__main__':
    main()
