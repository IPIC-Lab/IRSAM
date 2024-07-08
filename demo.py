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


def main(valid_datasets, args):
    # --- Step 1: Valid dataset ---
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
    if args.eval:
        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(args.restore_model))
            else:
                net.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        evaluate(net, valid_dataloaders)

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

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=net.device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            masks, edges = net(batched_input)

            torch.cuda.synchronize()

            IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

            FA, PD = Pd_Fa.get(len(valid_dataloader))
            _, IoU = IoU_metric.get()
            _, nIoU = nIoU_metric.get()

            tbar.set_description('IoU:%f, nIoU:%f, PD:%.8lf, FA:%.8lf'
                                 % (IoU, nIoU, PD[0], FA[0]))
            
        metric['iou'] = IoU
        metric['niou'] = nIoU
        metric['pd'] = PD[0]
        metric['fa'] = FA[0]
    return metric


if __name__ == "__main__":
    # --------------- Configuring the Valid datasets ---------------
    dataset_val_nuaa = {"name": "Sirstv2_512",
                        "im_dir": "datasets/Sirstv2_512/test_images",
                        "gt_dir": "datasets/Sirstv2_512/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_val_NUDT = {"name": "NUDT",
                        "im_dir": "datasets/NUDT-SIRST00/test_images",
                        "gt_dir": "datasets/NUDT-SIRST00/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_val_IRSTD = {"name": "IRSTD",
                         "im_dir": "datasets/IRSTD-1k/test_images",
                         "gt_dir": "datasets/IRSTD-1k/test_masks",
                         "im_ext": ".png",
                         "gt_ext": ".png"}

    valid_datasets = [dataset_val_nuaa]

    args = get_args_parser()

    main(valid_datasets, args)
