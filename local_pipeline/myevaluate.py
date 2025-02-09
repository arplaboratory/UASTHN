import numpy as np
import os
import torch
import argparse
from model.network import UASTHN
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
from sklearn.metrics import roc_curve, roc_auc_score

def load_model(args, model):
    if args.first_stage_ue and args.ue_method == "ensemble":
        for i in range(len(model.netG_list)):
            model_med = torch.load(model.ensemble_model_names[i], map_location='cuda:0')
            for key in list(model_med['netG'].keys()):
                model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
            for key in list(model_med['netG'].keys()):
                if key.startswith('module'):
                    del model_med['netG'][key]
            model.netG_list[i].load_state_dict(model_med['netG'], strict=True)
    else:
        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=True)
    if args.two_stages:
        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG_fine'].keys()):
            model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
        for key in list(model_med['netG_fine'].keys()):
            if key.startswith('module'):
                del model_med['netG_fine'][key]
        model.netG_fine.load_state_dict(model_med['netG_fine'], strict=True)
    
    model.setup() 
    if args.first_stage_ue and args.ue_method == "ensemble":
        for i in range(len(model.netG_list)):
            model.netG_list[i].eval()
    else:
        model.netG.eval()
    if args.two_stages:
        model.netG_fine.eval()
    return model

def test(args, wandb_log):
    if not args.identity:
        if not args.ue_method == "augment_ensemble":
            model = UASTHN(args)
            model = load_model(args, model)
        else:
            model = []
            args.ue_method = "augment"
            model_single = UASTHN(args)
            model_single = load_model(args, model_single)
            model.append(model_single)
            args.ue_method = "ensemble"
            model_single = UASTHN(args)
            model_single = load_model(args, model_single)
            model.append(model_single)
            args.ue_method = "augment_ensemble"
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
    total_ue_mask = torch.empty(0, len(args.ue_rej_std))
    total_ue_value = torch.empty(0)
    timeall=[]
    if args.generate_test_pairs:
        test_pairs = torch.zeros(len(val_dataset.dataset), dtype=torch.long)

    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        img1, img2, flow_gt,  H, query_utm, database_utm, index, pos_index  = [x for x in data_blob]
        if args.generate_test_pairs:
            test_pairs[index] = pos_index

        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")

        if not args.identity:
            if not args.ue_method == "augment_ensemble":
                model.set_input(img1, img2, flow_gt)
            else:
                for model_single in model:
                    model_single.set_input(img1, img2, flow_gt)
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        if not args.identity:
            with torch.no_grad():
                if not args.ue_method == "augment_ensemble":
                    time_start = time.time()
                    model.forward(for_test=True)
                    time_end = time.time()
                    timeall.append(time_end-time_start)
                    print(time_end-time_start)
                    four_pred = model.four_pred
                
                else:
                    time_start = time.time()
                    for model_single in model:
                        model_single.forward(for_test=True)
                    time_end = time.time()
                    timeall.append(time_end-time_start)
                    four_pred = model[0].four_pred
        else:
            four_pred = torch.zeros((flow_gt.shape[0], 2, 2, 2))

        mace_ = (flow_4cor - four_pred.cpu().detach())**2
        mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
        mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
        # print(mace_[0,:])
        ue_mask = torch.ones((mace_vec.shape[0], len(args.ue_rej_std)))
        if args.first_stage_ue:
            if args.ue_method == "augment_ensemble":
                model_eval = model[0]
                if args.ue_combine == "min":
                    model_eval.std_four_pred_five_crops = torch.min(torch.stack([model_eval.std_four_pred_five_crops, model[1].std_four_pred_five_crops], dim=-1), dim=-1)[0]
                elif args.ue_combine == "add":
                    model_eval.std_four_pred_five_crops = model_eval.std_four_pred_five_crops + model[1].std_four_pred_five_crops
                elif args.ue_combine == "max":
                    model_eval.std_four_pred_five_crops = torch.max(torch.stack([model_eval.std_four_pred_five_crops, model[1].std_four_pred_five_crops], dim=-1), dim=-1)[0]
                else:
                    raise NotImplementedError()
            else:
                model_eval = model
            ue_std = model_eval.std_four_pred_five_crops.view(model_eval.std_four_pred_five_crops.shape[0], -1)
            ue_mask_list = []
            if args.ue_std_method == "any":
                ue_value = torch.max(ue_std, dim=1)[0].cpu()
            elif args.ue_std_method == "all":
                ue_value = torch.min(ue_std, dim=1)[0].cpu()
            elif args.ue_std_method == "mean":
                ue_value = torch.mean(ue_std, dim=1).cpu()
            else:
                raise NotImplementedError()
            for j in range(len(args.ue_rej_std)):
                if args.ue_std_method == "any":
                    ue_mask_rej = torch.any(ue_std > args.ue_rej_std[j], dim=1).cpu()
                elif args.ue_std_method == "all":
                    ue_mask_rej = torch.all(ue_std > args.ue_rej_std[j], dim=1).cpu()
                elif args.ue_std_method == "mean":
                    ue_mask_rej = (torch.mean(ue_std, dim=1) > args.ue_rej_std[j]).cpu()
                else:
                    raise NotImplementedError()
                ue_mask = ~ue_mask_rej
                ue_mask_list.append(ue_mask)
            ue_mask = torch.stack(ue_mask_list, dim=1)
        else:
            model_eval = model
        total_ue_mask = torch.cat([total_ue_mask, ue_mask], dim=0)
        total_ue_value = torch.cat([total_ue_value, ue_value], dim=0)
        total_mace = torch.cat([total_mace,mace_vec], dim=0)
        
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
        total_ce = torch.cat([total_ce, ce_vec], dim=0)
        
        # if args.vis_all:
        #     save_dir = os.path.join(args.save_dir, 'vis')
        #     if not os.path.exists(save_dir):
        #         os.mkdir(save_dir)
        #     if not args.two_stages:
        #         save_overlap_bbox_img(model_eval.image_1, model_eval.real_warped_image_2, save_dir + f'/train_overlap_bbox_{i_batch}.png', four_point_gt, four_point_1, ue_mask=ue_mask)
        #     else:
        #         four_point_org_single_ori = torch.zeros((1, 2, 2, 2))
        #         four_point_org_single_ori[:, :, 0, 0] = torch.Tensor([0, 0])
        #         four_point_org_single_ori[:, :, 0, 1] = torch.Tensor([args.database_size - 1, 0])
        #         four_point_org_single_ori[:, :, 1, 0] = torch.Tensor([0, args.database_size - 1])
        #         four_point_org_single_ori[:, :, 1, 1] = torch.Tensor([args.database_size - 1, args.database_size - 1])
        #         four_point_bbox = model_eval.flow_bbox.cpu().detach() + four_point_org_single_ori
        #         alpha = args.database_size / args.resize_width
        #         four_point_bbox = four_point_bbox.flatten(2).permute(0, 2, 1).contiguous() / alpha
        #         save_overlap_bbox_img(model_eval.image_1, model_eval.real_warped_image_2, save_dir + f'/train_overlap_bbox_{i_batch}.png', four_point_gt, four_point_1, crop_bbox=four_point_bbox, ue_mask=ue_mask)
        #         if args.ue_method == "augment":
        #             four_point_gt_multi = four_point_gt.repeat(args.ue_num_crops, 1, 1)
        #             four_point_1_multi = four_point_1.repeat(args.ue_num_crops, 1, 1)
        #             save_overlap_bbox_img(model_eval.image_1_multi, model_eval.fake_warped_image_2_multi_before, save_dir + f'/train_overlap_bbox_before_recover_{i_batch}.png', four_point_gt_multi, four_point_1_multi)
        #             save_overlap_bbox_img(model_eval.image_1_multi, model_eval.fake_warped_image_2_multi_after, save_dir + f'/train_overlap_bbox_after_recover_{i_batch}.png', four_point_gt_multi, four_point_1_multi)
    for j in range(total_ue_mask.shape[1]):
        ue_mask_single = total_ue_mask[:,j].bool()
        final_ue_mask = torch.count_nonzero(ue_mask_single)/len(ue_mask_single)
        final_mace = torch.mean(total_mace[ue_mask_single]).item()
        final_ce = torch.mean(total_ce[ue_mask_single]).item()
        logging.info(f"MACE Metric {j}: {final_mace}")
        logging.info(f'CE Metric {j}: {final_ce}')
        logging.info(f'Success rate {j}:{final_ue_mask}')
        print(f"MACE Metric {j}: {final_mace}")
        print(f'CE Metric {j}: {final_ce}')
        print(f'Success rate {j}:{final_ue_mask}')
        if wandb_log:
            wandb.log({f"test_mace_{j}": final_mace})
            wandb.log({f"test_ce_{j}": final_ce})
            wandb.log({f"success_rate_{j}": final_ue_mask})
    # plot ROC using ue_value for MACE > 25.0 m
    y_label_0 = torch.zeros(len(total_mace)).long()
    y_label_0[total_mace > 4.167] = 1
    fpr0, tpr0, _ = roc_curve(y_label_0, total_ue_value)
    auc0 = roc_auc_score(y_label_0, total_ue_value)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr0, tpr0, 'g', label = 'MACE > 25.0 m: AUC = %0.2f' % auc0)
    plt.title('STHN two-stage')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(args.save_dir + "/ROC.png", bbox_inches='tight')
    plt.close()
    np.save(args.save_dir + "/fpr0.npy", fpr0)
    np.save(args.save_dir + "/tpr0.npy", tpr0)

    plt.figure(figsize=(8, 8))
    plt.hist(x, density=True, bins=30)  # density=False would make counts
    plt.ylabel('Frequency')
    plt.xlabel('Data')
    plt.savefig(args.save_dir + "/hist.png", bbox_inches='tight')
    plt.close()
    
    logging.info(np.mean(np.array(timeall[1:-1])))
    io.savemat(args.save_dir + '/resmat', {'matrix': total_mace.numpy()})
    np.save(args.save_dir + '/resnpy.npy', total_mace.numpy())
    plot_hist_helper(args.save_dir)
    if args.generate_test_pairs:
        torch.save(test_pairs, f"cache/{val_dataset.dataset.split}_{args.val_positive_dist_threshold}_pairs.pth")

if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if args.identity:
        args.save_dir = join(
        "test",
        args.save_dir,
        "identity",
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
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
        wandb.init(project="UASTHN-eval", entity="xjh19971", config=vars(args))
    test(args, wandb_log)