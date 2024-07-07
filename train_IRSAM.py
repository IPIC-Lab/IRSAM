# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as ops
from tqdm import tqdm
import argparse
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms  as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from thop import profile

from segment_anything_training.build_IRSAM import build_sam_IRSAM

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.metric import PD_FA, ROCMetric
from utils.loss_mask import DICE_loss
from utils.log import initialize_logger
import utils.misc as misc

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model_type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--no_prompt_checkpoint", type=str, default=None,
                        help="The path to the SAM checkpoint trained with no prompt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=1001, type=int)
    parser.add_argument('--dataloader_size', default=[512, 512], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=10, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(train_datasets, valid_datasets, args):
    # --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets= create_dataloaders(train_im_gt_list,
                                                               my_transforms=[
                                                                   RandomHFlip(),
                                                                   LargeScaleJitter()
                                                               ],
                                                               batch_size=args.batch_size_train,
                                                               training=True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.dataloader_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- Step 2: Load pretrained Network---
    net = build_sam_IRSAM(checkpoint=args.checkpoint)
    if torch.cuda.is_available():
        net.cuda()

    # --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.lr_drop_epoch, gamma=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(args.restore_model))
            else:
                net.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        evaluate(net, valid_dataloaders)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    os.makedirs(args.output, exist_ok=True)

    best_iou = 0
    best_nIoU = 0
    best_FA = 1000000000000000
    best_PD = 0

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    BCEloss = nn.BCELoss()
    Diceloss = DICE_loss

    net.train()
    _ = net.to(device=args.device)

    for epoch in range(epoch_start, epoch_num):
        net.train()
        losses_bce = []
        losses_dice = []
        losses_iou = []

        tbar = tqdm(train_dataloaders)
        for data in tbar:
            inputs, labels, edges_gt = data['image'], data['label'], data['edge']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            masks, edges = net(batched_input)

            masks = masks.cpu()
            edges = edges.cpu()
            labels = labels.cpu()

            loss_bce = BCEloss(edges, edges_gt/255.)
            loss_dice, loss_iou = Diceloss(masks, labels / 255.)
            loss = 5 * loss_bce + loss_iou

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_bce.append(loss_bce.detach())
            losses_dice.append(loss_dice.detach())
            losses_iou.append(loss_iou.detach())
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f, bce_loss:%f, dice_loss:%f, iou_loss:%f'
                                 % (epoch, optimizer.param_groups[0]['lr'],
                                    10 * np.mean(losses_bce) + np.mean(losses_dice) + np.mean(losses_iou),
                                    np.mean(losses_bce), np.mean(losses_dice), np.mean(losses_iou)))
        lr_scheduler.step(epoch)
        metric = evaluate(net, valid_dataloaders)

        if metric['iou'] > best_iou:
            if os.path.exists(ops.join(args.output, "Best_IoU-%.4f.pth" % best_iou)):
                os.remove(ops.join(args.output, "Best_IoU-%.4f.pth" % best_iou))
            best_iou = metric['iou']
            torch.save(net.state_dict(), ops.join(args.output, "Best_IoU-%.4f.pth" % best_iou))
        if metric['niou'] > best_nIoU:
            if os.path.exists(ops.join(args.output, "Best_nIoU-%.4f.pth" % best_nIoU)):
                os.remove(ops.join(args.output, "Best_nIoU-%.4f.pth" % best_nIoU))
            best_nIoU = metric['niou']
            torch.save(net.state_dict(), ops.join(args.output, "Best_nIoU-%.4f.pth" % best_nIoU))
        if metric['fa'] * 1000000 < best_FA:
            best_FA = metric['fa'] * 1000000
        if metric['pd'] > best_PD:
            best_PD = metric['pd']

        if epoch % args.model_save_fre == 0:
                model_name = "/epoch_" + str(epoch)+"IoU-%.4f.pth" % metric['iou']
                misc.save_on_master(net.state_dict(), args.output + model_name)

    # Finish training
    print('Best IoU:%f, Best nIoU:%f, Best PD:%.8lf, Best FA:%.8lf' % (best_iou, best_nIoU, best_PD, best_FA))
    print("Training Reaches The Maximum Epoch Number")


def evaluate(net, valid_dataloaders):
    net.eval()
    metric = dict()

    IoU_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)

    ROC = ROCMetric(1, 10)
    Pd_Fa = PD_FA(1, 10)

    IoU_metric.reset()
    nIoU_metric.reset()
    Pd_Fa.reset()
    for k in range(len(valid_dataloaders)):
        valid_dataloader = valid_dataloaders[k]

        tbar = tqdm(valid_dataloader)
        for data_val in tbar:
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val[
                'label'], data_val['shape'], data_val['ori_label']
            # plt.subplot(2, 3, 1)
            # plt.imshow(inputs_val[0].permute(1, 2, 0).cpu().numpy()/255.)
            # plt.subplot(2, 3, 2)
            # plt.imshow(labels_val[0].permute(1, 2, 0).cpu().numpy() / 255.)

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()

            # plt.subplot(2, 3, 3)
            # plt.imshow((prompt_masks[0] > 0).permute(1, 2, 0).cpu().numpy() / 255.)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=net.device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            # starter.record()
            masks, edges = net(batched_input)
            # flops, params = profile(net, inputs=(batched_input,))
            # print(flops, params)
            # ender.record()

            torch.cuda.synchronize()

            # curr_time = starter.elapsed_time(ender)
            # timings[i] = curr_time
            # i += 1


            # plt.subplot(2, 3, 4)
            # plt.imshow((masks[0]>0).permute(1, 2, 0).cpu().numpy() / 255.)
            # plt.subplot(2, 3, 5)
            # plt.imshow((masks_hq[0]>0).permute(1, 2, 0).cpu().numpy() / 255.)
            #
            # plt.show()

            IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            # ROC.update(IoU_img[0].cpu(), labels_ori[0].cpu())
            Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

            FA, PD = Pd_Fa.get(len(valid_dataloader))
            _, IoU = IoU_metric.get()
            _, nIoU = nIoU_metric.get()

            tbar.set_description('IoU:%f, nIoU:%f, PD:%.8lf, FA:%.8lf'
                                 % (IoU, nIoU, PD[0], FA[0]))

        # avg = timings.sum() / 86
        #
        # print(timings)
        # print('\navg={}\n'.format(avg))

        metric['iou'] = IoU
        metric['niou'] = nIoU
        metric['pd'] = PD[0]
        metric['fa'] = FA[0]
    return metric


if __name__ == "__main__":
    # --------------- Configuring the Train and Valid datasets ---------------
    dataset_train_nuaa = {"name": "Sirstv2_512",
                          "im_dir": "datasets/Sirstv2_512/train_images",
                          "gt_dir": "datasets/Sirstv2_512/train_masks",
                          "im_ext": ".png",
                          "gt_ext": ".png"}

    dataset_val_nuaa = {"name": "Sirstv2_512",
                        "im_dir": "datasets/Sirstv2_512/test_images",
                        "gt_dir": "datasets/Sirstv2_512/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_train_NUDT = {"name": "NUDT",
                          "im_dir": "datasets/NUDT-SIRST00/train_images",
                          "gt_dir": "datasets/NUDT-SIRST00/train_masks",
                          "im_ext": ".png",
                          "gt_ext": ".png"}

    dataset_val_NUDT = {"name": "NUDT",
                        "im_dir": "datasets/NUDT-SIRST00/test_images",
                        "gt_dir": "datasets/NUDT-SIRST00/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_train_IRSTD = {"name": "IRSTD",
                           "im_dir": "datasets/IRSTD-1k/train_images",
                           "gt_dir": "datasets/IRSTD-1k/train_masks",
                           "im_ext": ".png",
                           "gt_ext": ".png"}

    dataset_val_IRSTD = {"name": "IRSTD",
                         "im_dir": "datasets/IRSTD-1k/test_images",
                         "gt_dir": "datasets/IRSTD-1k/test_masks",
                         "im_ext": ".png",
                         "gt_ext": ".png"}

    train_datasets = [dataset_train_nuaa]
    valid_datasets = [dataset_val_nuaa]

    args = get_args_parser()

    main(train_datasets, valid_datasets, args)
