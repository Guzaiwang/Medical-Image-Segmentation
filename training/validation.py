import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.distributed as dist
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice, calculate_dice_split, calculate_dice_binary
import numpy as np
from numpy import *
from .utils import concat_all_gather, remove_wrap_arounds
import cv2
import logging
import pdb
from utils import is_master
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image

import sklearn.metrics as metrics


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    return acc, sen


def mask_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)  # I assume this is faster as mask1 == 1 is a bool array
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou


def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()

    label_1D = label_1D
    auc = metrics.roc_auc_score(label_1D.astype(np.uint8), result_1D)

    return auc

def dice(im1, im2, empty_score=1.0):
    """
    This code is from https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum



def validation(net, dataloader, args):
    
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    for i in range(args.classes-1): # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])

    inference = get_inference(args)
    
    logging.info("Evaluating")

    with torch.no_grad():
        iterator = tqdm(dataloader)
        for (images, labels) in iterator:
            # spacing here is used for distance metrics calculation
            
            inputs, labels = images.float().cuda(), labels.cuda().to(torch.int8)
            
            # if args.dimension == '2d':
            #     inputs = inputs.permute(1, 0, 2, 3)
            
            pred = inference(net, inputs, args)

            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.to(torch.int8)

            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
               

            # tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging (HD and ASD computation for large 3D images is slow)
            #tmp_ASD_list = np.zeros(args.classes-1)
            #tmp_HD_list = np.zeros(args.classes-1)

            # tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            # tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)
        
            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM.
            # Use calculate_dice_split instead if got OOM, it will evaluate patch by patch to reduce gpu memory consumption.
            dice, _, _ = calculate_dice_binary(label_pred.view(-1, 1), labels.view(-1, 1), 2)
            # dice, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            # exclude background
            # dice = dice.cpu().numpy()[1:]
            dice = dice.cpu().numpy()

            unique_cls = torch.unique(labels)
            for cls in range(0, args.classes-1):
                if cls+1 in unique_cls: 
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    # ASD_list[cls].append(tmp_ASD_list[cls])
                    # HD_list[cls].append(tmp_HD_list[cls])
                    dice_list[cls].append(dice)

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append(np.array(dice_list[cls]).mean())
        out_ASD.append(np.array(dice_list[cls]).mean())
        out_HD.append(np.array(dice_list[cls]).mean())
        # out_ASD.append(np.array(ASD_list[cls]).mean())
        # out_HD.append(np.array(HD_list[cls]).mean())

    return np.array(out_dice), np.array(out_ASD), np.array(out_HD)

def validation2d(net, dataset, args):
    
    net.eval()

    inference = get_inference(args)
    
    logging.info("Evaluating")

    # print(dataset.img_path_list)
    # print(dataset.msk_path_list)
    with torch.no_grad():
        auc_list = []
        acc_list = []
        sen_list = []
        iou_list = []
        dice_list = []
        for imd_idx in range(len(dataset.img_path_list)):
            img = cv2.imread(dataset.img_path_list[imd_idx])
            img = cv2.resize(img, (448, 448))
            label = np.array(Image.open(dataset.msk_path_list[imd_idx]))
            if not args.test_aug:
                # without test augmentation
                img = img[None].transpose(0, 3, 1, 2)
                img = img / 255.0 * 3.2 - 1.6
                threshold = 0.5
                inputs = V(torch.Tensor(img).cuda())
                pred = inference(net, inputs, args)
                pred[pred > threshold] = 1
                pred[pred <= threshold] = 0
                label_pred = pred.to(torch.int8)
            label_pred = label_pred.squeeze().cpu().data.numpy().astype(np.float32)
            label_pred = cv2.resize(label_pred, (np.shape(label)[1], np.shape(label)[0]))
            label_pred[label_pred > threshold] = 1
            label_pred[label_pred <=threshold] = 0
            auc = calculate_auc_test(label_pred, label)
            acc, sen = accuracy(label_pred, label)
            iou_loss = 1 - mask_iou(label_pred, label)
            dice_loss = dice(label_pred, label)
            auc_list.append(auc)
            acc_list.append(acc)
            sen_list.append(sen)
            iou_list.append(iou_loss)
            dice_list.append(dice_loss)
        print("auc value is", mean(auc_list))
        print("acc value is", mean(acc_list))
        print("sen value is", mean(sen_list))
        print("iou_loss value is", mean(iou_list))
        print("dice value is", mean(dice_list))

def validation_ddp(net, dataloader, args):
    
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    unique_labels_list = []

    inference = get_inference(args)

    logging.info(f"Evaluating")

    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for (images, labels, spacing) in iterator:
            # spacing here is used for distance metrics calculation
            
            inputs, labels = images.cuda(args.proc_idx).float(), labels.cuda(args.proc_idx).long()
            
            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)
            
            pred = inference(net, inputs, args)

            _, label_pred = torch.max(pred, dim=1)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
 

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging. (HD and ASD computation for large 3D images are slow)
            #tmp_ASD_list = np.zeros(args.classes-1)
            #tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM. Put tensors to cpu if needed.
            tmp_dice_list, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            #tmp_dice_list, _, _ = calculate_dice(label_pred.view(-1, 1).cpu(), labels.view(-1, 1).cpu(), args.classes)


            unique_labels = torch.unique(labels).cpu().numpy()
            unique_labels =  np.pad(unique_labels, (100-len(unique_labels), 0), 'constant', constant_values=0)
            # the length of padding is just a randomly picked number (most medical tasks don't have over 100 classes)
            # The padding here is because the all_gather in DDP requires the tensors in gpus have the same shape

            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)
            tmp_ASD_list = np.expand_dims(tmp_ASD_list, axis=0)
            tmp_HD_list = np.expand_dims(tmp_HD_list, axis=0)

            if args.distributed:
                # gather results from all gpus
                tmp_dice_list = concat_all_gather(tmp_dice_list)
                
                unique_labels = torch.from_numpy(unique_labels).cuda()
                unique_labels = concat_all_gather(unique_labels)
                unique_labels = unique_labels.cpu().numpy()
                
                tmp_ASD_list = torch.from_numpy(tmp_ASD_list).cuda()
                tmp_ASD_list = concat_all_gather(tmp_ASD_list)
                tmp_ASD_list = tmp_ASD_list.cpu().numpy()

                tmp_HD_list = torch.from_numpy(tmp_HD_list).cuda()
                tmp_HD_list = concat_all_gather(tmp_HD_list)
                tmp_HD_list = tmp_HD_list.cpu().numpy()


            tmp_dice_list = tmp_dice_list.cpu().numpy()[:, 1:] # exclude background
            for idx in range(len(tmp_dice_list)):  # get the result for each sample
                ASD_list.append(tmp_ASD_list[idx])
                HD_list.append(tmp_HD_list[idx])
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])
    
    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(dataloader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        for _ in range(padding_size):
            ASD_list.pop()
            HD_list.pop()
            dice_list.pop()
            unique_labels_list.pop()
    

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append([])
        out_ASD.append([])
        out_HD.append([])

    for idx in range(len(dice_list)):
        for cls in range(0, args.classes-1):
            if cls+1 in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
                out_ASD[cls].append(ASD_list[idx][cls])
                out_HD[cls].append(HD_list[idx][cls])
    
    out_dice_mean, out_ASD_mean, out_HD_mean = [], [], []
    for cls in range(0, args.classes-1):
        out_dice_mean.append(np.array(out_dice[cls]).mean())
        out_ASD_mean.append(np.array(out_ASD[cls]).mean())
        out_HD_mean.append(np.array(out_HD[cls]).mean())

    return np.array(out_dice_mean), np.array(out_ASD_mean), np.array(out_HD_mean)


