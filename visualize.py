#!/usr/bin/python3
# -*- mode: python; python-indent-offset: 4 -*-
import numpy as np
import torchvision
import time
import os
import shutil
import copy
import logging
import pdb
import time
import argparse

import sys
import cv2

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
logger = logging.getLogger('run_logger')

def horizontal_aligned(a, b):
    v_diff = abs((a[1] + a[0]) - (b[1] + b[0])) / 2
    return v_diff < max((a[1] - a[0]) / 2, (b[1] - b[0]) / 2)

def group_by_y(letters):
    """letters is an array of (letter, left-topx, left-topy, right-bottomx, right-bottomy)"""
    groups = []
    for l in sorted(letters, key=lambda x: x[1]):
        found = False
        for g in groups:
            logger.debug("l:{}, g:{}, horizontal aligned:{}".format(l, g[-1], horizontal_aligned((g[-1][2], g[-1][4]), (l[2], l[4])))) 
            if (horizontal_aligned((g[-1][2], g[-1][4]), (l[2], l[4]))):
                g.append(l)
                found = True
                break
        if (not found):
            groups.append([l])
    for g in groups:
        logger.debug("Group: l{} {}".format(len(g), ''.join([str(s[0]) for s in g])))
    return groups


def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

    
def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    if b[1] <= 10:
        y = b[3] + 11
    else:
        y = b[1] - 1
    cv2.putText(image, caption, (b[0], y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#    cv2.putText(image, caption, (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color=(0,0,255), thickness=1)


def group_and_filter_boxes(boxes):
    """Group boxes by y cooridates and filter overlapped ones"""
    bb_groups = group_by_y(boxes)
    bb_groups_filtered = []
    for g in bb_groups:
        if len(g) <= 1:
            bb_groups_filtered.append(g)
            continue
        g_filtered = [g[0]]
        pre_bb = g[0]
        for bb in g[1:]:
            iou = bb_iou(pre_bb[1:], bb[1:])
            if iou > 0.25:
                continue
            g_filtered.append(bb)
            pre_bb = bb
        bb_groups_filtered.append(g_filtered)
    return bb_groups_filtered


def draw_boxes(img, bb_groups):
    for g in bb_groups:
        for b in g:
            draw_caption(img, b[1:], b[0])


def write_results(logf, results):
    results.sort(key=lambda x: x[2])
    with open(logf, "w") as lf:
        for r in results:
            lf.write("{:30s}, {:30s}, {}\n".format(r[0], r[1], r[2]))
            
                         
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='csv')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument('--output', help='Output directory of images with boxes.')

    parser = parser.parse_args(args)
    
    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    if os.path.exists(parser.output):
        shutil.rmtree(parser.output)
    os.makedirs(os.path.join(parser.output, "pass"))
    os.makedirs(os.path.join(parser.output, "fail"))
                    

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = UnNormalizer()

    matched = 0
    not_matched = 0
    failed = []
    passed = []
    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            elasped = time.time()
            idxs = np.where(scores.cpu()>0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255
            
            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            bb_found = []
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                bb_found.append([label_name, x1, y1, x2, y2])

            bb_groups_filtered = group_and_filter_boxes(bb_found)
            draw_boxes(img, bb_groups_filtered)
            
            _, r, _ = data['annot'].shape
            labels_truth = []
            for i in range(r):
                labels_truth.append([int(data['annot'][0, i, -1])] + [int(d) for d in data['annot'][0, i, :-1]])

            bb_group_truth = group_by_y(labels_truth)
            strings_found = '-'.join([''.join([str(c[0]) for c in l]) for l in bb_groups_filtered])
            strings_truth = '-'.join([''.join([str(c[0]) for c in l]) for l in bb_group_truth])
            img_name = Path(data['image_name'][0]).name
            if strings_truth == strings_found:
                result = 'pass'
                matched += 1
                passed.append((strings_found, strings_truth, img_name))
            else:
                result = 'fail'
                not_matched += 1
                failed.append((strings_found, strings_truth, img_name))
            total = matched + not_matched
            print("{:15s}: Found: {}, Truth: {}, {}, {}/{}={:5.2f}".format(
                result, strings_found, strings_truth, img_name, matched, total, matched / float(total))) 
            
            cv2.imwrite(os.path.join(parser.output, result, img_name), img)
    write_results(os.path.join(parser.output, "fail.log"), failed)
    write_results(os.path.join(parser.output, "pass.log"), passed)
    
if __name__ == '__main__':
    main()
