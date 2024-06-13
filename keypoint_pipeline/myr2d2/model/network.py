import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox
from model.patchnet import Quad_L2Net_ConfCFS
from model.losses import MultiLoss
from model.sampler import NghSampler2
from utils import coords_grid, fetch_optimizer, warp
import os
import sys
from model.sync_batchnorm import convert_model
import wandb
import torchvision
import random
import time
import logging
import numpy as np

autocast = torch.cuda.amp.autocast

class KeyNet():
    def __init__(self, args, for_training=False):
        super().__init__()
        self.args = args
        self.device = args.device
        self.four_point_org_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_single[:, :, 0, 1] = torch.Tensor([self.args.resize_width - 1, 0]).to(self.device)
        self.four_point_org_single[:, :, 1, 0] = torch.Tensor([0, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_single[:, :, 1, 1] = torch.Tensor([self.args.resize_width - 1, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_large_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_large_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 0, 1] = torch.Tensor([self.args.database_size - 1, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 0] = torch.Tensor([0, self.args.database_size - 1]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 1] = torch.Tensor([self.args.database_size - 1, self.args.database_size - 1]).to(self.device) # Only to calculate flow so no -1
        self.netG = Quad_L2Net_ConfCFS()
        self.shift_flow_bbox = None
        self.sampler = NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)
        self.critierion_AUX = MultiLoss(1, ReliabilityLoss(self.sampler, base=0.5, nq=20),
                                        1, CosimLoss(N=self.args.N),
                                        1, PeakyLoss(N=self.args.N))
        if for_training:
            self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()))
            
    def setup(self):
        self.netG = self.init_net(self.netG)

    def init_net(self, model):
        # model = torch.nn.DataParallel(model)
        # if torch.cuda.device_count() >= 2:
        #     # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        #     model = convert_model(model)
        #     model = model.to(self.device)
        model = model.to(self.device)
        return model
    
    def set_input(self, A, B, flow_gt=None, neg_A=None):
        self.image_1_ori = A.to(self.device, non_blocking=True)
        self.image_2 = B.to(self.device, non_blocking=True)
        self.flow_gt = flow_gt.to(self.device, non_blocking=True)
        if self.flow_gt is not None:
            # self.real_warped_image_2 = mywarp(self.image_2, self.flow_gt, self.four_point_org_single) # Comment for performance evaluation 
            self.flow_4cor = torch.zeros((self.flow_gt.shape[0], 2, 2, 2)).to(self.flow_gt.device)
            self.flow_4cor[:, :, 0, 0] = self.flow_gt[:, :, 0, 0]
            self.flow_4cor[:, :, 0, 1] = self.flow_gt[:, :, 0, -1]
            self.flow_4cor[:, :, 1, 0] = self.flow_gt[:, :, -1, 0]
            self.flow_4cor[:, :, 1, 1] = self.flow_gt[:, :, -1, -1]
        else:
            self.real_warped_image_2 = None
        self.image_1 = F.interpolate(self.image_1_ori, size=self.args.resize_width, mode='bilinear', align_corners=True, antialias=True)
        if neg_A is not None:
            self.image_1_neg_ori = neg_A.to(self.device, non_blocking=True)
            self.image_1_neg = F.interpolate(self.image_1_neg_ori, size=self.args.resize_width, mode='bilinear', align_corners=True, antialias=True)
        else:
            self.image_1_neg = None
        
    def forward(self, for_training=False, for_test=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # time1 = time.time()
        self.output1 = self.netG(self.image_1)
        self.output2 = self.netG(self.image_2)
        # time2 = time.time()
        # logging.debug("Time for 1st forward pass: " + str(time2 - time1) + " seconds")
        # self.fake_warped_image_2 = mywarp(self.image_2, self.four_pred, self.four_point_org_single) # Comment for performance evaluation

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.four_pred, self.flow_gt, self.args.gamma, self.args, self.metrics) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Homo
        self.metrics["G_loss"] = self.loss_G.cpu().item()
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward(for_training=True) # Calculate Fake A
        if self.args.neg_training:
            self.forward_neg(for_training=True)
        self.metrics = dict()
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        nn.utils.clip_grad_norm_(self.netG.parameters(), self.args.clip)
        self.optimizer_G.step()             # update G's weights
        return self.metrics

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler_G.step()

def mywarp(x, flow_pred, four_point_org_single, ue_std=None):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if not torch.isnan(flow_pred).any():
        if flow_pred.shape[-1] != 2:
            flow_4cor = torch.zeros((flow_pred.shape[0], 2, 2, 2)).to(flow_pred.device)
            flow_4cor[:, :, 0, 0] = flow_pred[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_pred[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_pred[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_pred[:, :, -1, -1]
        else:
            flow_4cor = flow_pred

        four_point_1 = flow_4cor + four_point_org_single
        
        four_point_org = four_point_org_single.repeat(flow_pred.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
        four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous() 
        try:
            H = tgm.get_perspective_transform(four_point_org, four_point_1)
        except Exception:
            logging.debug("No solution")
            H = torch.eye(3).to(four_point_org.device).repeat(four_point_1.shape[0],1,1)
        warped_image = tgm.warp_perspective(x, H, (x.shape[2], x.shape[3]))
    else:
        logging.debug("Output NaN by model error.")
        warped_image = x
    return warped_image