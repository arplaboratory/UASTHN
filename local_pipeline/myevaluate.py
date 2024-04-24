import numpy as np
import os
import torch
import argparse
from model.network import UAGL
from utils import save_overlap_img, save_img, setup_seed, save_overlap_bbox_img
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
from tqdm import tqdm
import cv2
import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
from plot_hist import plot_hist_helper
import torch.nn.functional as F
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import wandb

def test(args, wandb_log):
    if not args.identity:
        model = UAGL(args)
        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=False)
        if args.two_stages:
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
            for key in list(model_med['netG_fine'].keys()):
                if key.startswith('module'):
                    del model_med['netG_fine'][key]
            model.netG_fine.load_state_dict(model_med['netG_fine'])
        
        model.setup() 
        model.netG.eval()
        if args.two_stages:
            model.netG_fine.eval()
    else:
        model = None
    if args.test:
        val_dataset = datasets.fetch_dataloader(args, split='test')
    else:
        val_dataset = datasets.fetch_dataloader(args, split='val')
    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)
    
def evaluate_SNet(model, val_dataset, batch_size=0, args = None, wandb_log=False):

    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)
    total_flow = torch.empty(0)
    total_ce = torch.empty(0)
    total_ue_mask = torch.empty(0)
    timeall=[]
    mace_conf_list = []
    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        img1, img2, flow_gt,  H, query_utm, database_utm  = [x for x in data_blob]

        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")

        if not args.identity:
            model.set_input(img1, img2, flow_gt)
            flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        if not args.identity:
            # time_start = time.time()
            model.forward()
            # time_end = time.time()
            four_pred = model.four_pred
            # timeall.append(time_end-time_start)
            # print(time_end-time_start)
        else:
            four_pred = torch.zeros((flow_gt.shape[0], 2, 2, 2))

        mace_ = (flow_4cor - four_pred.cpu().detach())**2
        mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
        mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
        # print(mace_[0,:])
        ue_mask = torch.ones_like(mace_vec)
        if args.first_stage_ue:
            ue_std = model.std_four_pred_five_crops.view(model.std_four_pred_five_crops.shape[0], -1)
            beta = 512 / args.resize_width
            if args.ue_std_method == "any":
                ue_mask = torch.any(ue_std > args.ue_rej_std / beta, dim=1).cpu()
            elif args.ue_std_method == "all":
                ue_mask = torch.all(ue_std > args.ue_rej_std / beta, dim=1).cpu()
            elif args.ue_std_method == "mean":
                ue_mask = (torch.mean(ue_std) > args.ue_rej_std / beta).cpu()
            else:
                raise NotImplementedError()
        total_ue_mask = torch.cat([total_ue_mask, ue_mask], dim=0)
        final_ue_mask = torch.count_nonzero(total_ue_mask)/len(total_ue_mask)

        total_mace = torch.cat([total_mace,mace_vec*ue_mask], dim=0)
        final_mace = torch.mean(total_mace).item()
        
        # CE
        four_point_org_single = torch.zeros((1, 2, 2, 2))
        four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0])
        four_point_org_single[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
        four_point_org_single[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
        four_point_org_single[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])
        four_point_1 = four_pred.cpu().detach() + four_point_org_single
        four_point_org = four_point_org_single.repeat(four_point_1.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
        four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
        four_point_gt = flow_4cor.cpu().detach() + four_point_org_single
        four_point_gt = four_point_gt.flatten(2).permute(0, 2, 1).contiguous()
        H = tgm.get_perspective_transform(four_point_org, four_point_1)
        center_T = torch.tensor([args.resize_width/2-0.5, args.resize_width/2-0.5, 1]).unsqueeze(1).unsqueeze(0).repeat(H.shape[0], 1, 1)
        w = torch.bmm(H, center_T).squeeze(2)
        center_pred_offset = w[:, :2]/w[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)
        # alpha = args.database_size / args.resize_width
        # center_gt_offset = (query_utm - database_utm).squeeze(1) / alpha
        # temp = center_gt_offset[:, 0].clone()
        # center_gt_offset[:, 0] = center_gt_offset[:, 1]
        # center_gt_offset[:, 1] = temp # Swap!
        H_gt = tgm.get_perspective_transform(four_point_org, four_point_gt)
        w_gt = torch.bmm(H_gt, center_T).squeeze(2)
        center_gt_offset = w_gt[:, :2]/w_gt[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)
        ce_ = (center_pred_offset - center_gt_offset)**2
        ce_ = ((ce_[:,0] + ce_[:,1])**0.5)
        ce_vec = ce_
        total_ce = torch.cat([total_ce, ce_vec*ue_mask], dim=0)
        final_ce = torch.mean(total_ce).item()
        
        if args.vis_all:
            save_dir = os.path.join(args.save_dir, 'vis')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not args.two_stages:
                save_overlap_bbox_img(model.image_1, model.fake_warped_image_2, save_dir + f'/train_overlap_bbox_{i_batch}.png', four_point_gt, four_point_1, ue_mask=ue_mask)
            else:
                four_point_org_single_ori = torch.zeros((1, 2, 2, 2))
                four_point_org_single_ori[:, :, 0, 0] = torch.Tensor([0, 0])
                four_point_org_single_ori[:, :, 0, 1] = torch.Tensor([args.database_size - 1, 0])
                four_point_org_single_ori[:, :, 1, 0] = torch.Tensor([0, args.database_size - 1])
                four_point_org_single_ori[:, :, 1, 1] = torch.Tensor([args.database_size - 1, args.database_size - 1])
                four_point_bbox = model.flow_bbox.cpu().detach() + four_point_org_single_ori
                alpha = args.database_size / args.resize_width
                four_point_bbox = four_point_bbox.flatten(2).permute(0, 2, 1).contiguous() / alpha
                save_overlap_bbox_img(model.image_1, model.fake_warped_image_2, save_dir + f'/train_overlap_bbox_{i_batch}.png', four_point_gt, four_point_1, crop_bbox=four_point_bbox, ue_mask=ue_mask)

    logging.info(f"MACE Metric: {final_mace}")
    logging.info(f'CE Metric: {final_ce}')
    logging.info(f'Success rate:{final_ue_mask}')
    print(f"MACE Metric: {final_mace}")
    print(f'CE Metric: {final_ce}')
    print(f'Success rate:{final_ue_mask}')
    if wandb_log:
        wandb.log({"test_mace": final_mace})
        wandb.log({"test_ce": final_ce})
        wandb.log({"success_rate": final_ue_mask})
    logging.info(np.mean(np.array(timeall[1:-1])))
    io.savemat(args.save_dir + '/resmat', {'matrix': total_mace.numpy()})
    np.save(args.save_dir + '/resnpy.npy', total_mace.numpy())
    plot_hist_helper(args.save_dir)

if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if args.identity:
        pass
    else:
        args.save_dir = join(
        "test",
        args.save_dir,
        args.eval_model.split("/")[-2],
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    logging.debug(args)
    wandb_log = True
    if wandb_log:
        wandb.init(project="UAGL-eval", entity="xjh19971", config=vars(args))
    test(args, wandb_log)