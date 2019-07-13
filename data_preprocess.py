'''Make image/labels/boxes from an annotation file.

This program will auto kill the image data which has no annotation.
The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...

Author : Wan-Cyuan Fan

'''
from __future__ import print_function

import os
import sys
import random
from tqdm import tqdm
import cv2
from retinanet import RetinaNet
from torch.autograd import Variable
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop

from pycocotools.coco import COCO
from class_cgf import *
import torchvision

parser = argparse.ArgumentParser(description='PyTorch data Preprocessing')
parser.add_argument('--dataType', default='train2017', type=str, help='dataType: train2017,val2017')
parser.add_argument('--build_annos','-ba', action='store_true', help='mode for build_annos')
parser.add_argument('--build_mPA_GT','-bg', action='store_true', help='mode for build_mPA_GT')
parser.add_argument('--test_image','-t', action='store_true', help='Testing mode ON and save the test image in ./')
args = parser.parse_args()


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    if order == 'xywh2xyxy':
        x = boxes[0] ; y = boxes[1] ; w = boxes[2] ; h = boxes[3]
        return [x,y,x+w,y+h] #[x-w/2.,y-h/2.,x+w/2.,y+h/2.]
    if order == 'xyxy2xywh':
        xm = boxes[0] ; ym = boxes[1] ; xM = boxes[2] ; yM = boxes[3]
        w = xM - xm
        h = yM - ym
        return [xm+w/2.,ym+h/2.,w,h]

def build_annos(root, ann_root,dataType):
    '''
    Args:
      root: (str) ditectory to images.
    '''
    print("<"+"="*20+">")
    print("[Building annotations]")
    fnames = os.listdir(root)
    fnames.sort()
    coco = COCO(ann_root)
    print("Total number of images : ",len(fnames))

    # save csv
    file = open("./data/%s.txt"%(dataType),"w")

    for i, name in enumerate(tqdm(fnames)):
        img_num = int(name.replace(".jpg",""))
        annIds = coco.getAnnIds(imgIds=[img_num], iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(anns) == 0:
            continue
        file.write("%s "%(name))
        for i, ann in enumerate(anns):
            coco_label = int(ann['category_id'])
            label = class_map(coco_label)
            # note that the order of BBox in COCO dataset is xywh where x and y is the up-left point
            # Not the center of BBox
            xywh = [float(ann['bbox'][0]),float(ann['bbox'][1]),float(ann['bbox'][2]),float(ann['bbox'][3])]
            bbox = change_box_order(xywh,'xywh2xyxy')

            file.write("%f %f %f %f %d "%(bbox[0],bbox[1],bbox[2],bbox[3],label))
        file.write("\n")
    print("[Done]")

    return True

def build_mPA_GT(root, ann_root,dataType):
    '''
    Args:
      root: (str) ditectory to images.
    '''
    print("<"+"="*20+">")
    print("[Building GT for mPA]")
    fnames = os.listdir(root)
    fnames.sort()
    coco = COCO(ann_root)
    print("Total number of images : ",len(fnames))

    for i, name in enumerate(tqdm(fnames)):
        img_num = int(name.replace(".jpg",""))
        annIds = coco.getAnnIds(imgIds=[img_num], iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(anns) == 0:
            continue
        file = open("./mPA/GT/%s.txt"%(name.replace(".jpg","")),"w")
        for i, ann in enumerate(anns):
            coco_label = int(ann['category_id'])
            label = class_map(coco_label)
            # note that the order of BBox in COCO dataset is xywh where x and y is the up-left point
            # Not the center of BBox
            xywh = [float(ann['bbox'][0]),float(ann['bbox'][1]),float(ann['bbox'][2]),float(ann['bbox'][3])]
            bbox = change_box_order(xywh,'xywh2xyxy')
            file.write("%s %.3f %.3f %.3f %.3f\n"%(my_cate[label],bbox[0],bbox[1],bbox[2],bbox[3]))
    print("[Done]")
    return True

def test(ku = False):

    dataType = args.dataType
    root = "../COCO_dataset/images/%s"%(dataType)
    ann_root = "../COCO_dataset/annotations/instances_%s.json"%(dataType)

    if args.build_annos:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        build_annos(root, ann_root,dataType)
    if args.build_mPA_GT:
        if not os.path.exists("./mPA"):
            os.makedirs("./mPA")
        if not os.path.exists("./mPA/GT"):
            os.makedirs("./mPA/GT")
        build_mPA_GT(root, ann_root,dataType)

    if args.test_image :
        fnames = []
        boxes = []
        labels = []

        with open("./data/%s.txt"%(dataType)) as f:

            lines = f.readlines()
        if ku == True:
            with open("./data/coco17_train.txt") as f:
                lines = f.readlines()
        for line in lines:
            splited = line.strip().split()
            fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            boxes.append(torch.Tensor(box))
            labels.append(torch.LongTensor(label))
        print("Total number of BBox : ",len(boxes))
        print("Total number of labels : ",len(labels))

        test_idx = 2200
        test_fname = fnames[test_idx]
        test_box = boxes[test_idx]
        test_label = labels[test_idx]

        img = Image.open(os.path.join(root,test_fname))

        draw = ImageDraw.Draw(img)

        for i,(box,label) in enumerate(zip(test_box,test_label)):
            draw.rectangle(list(box), outline=color_map(int(label)),width = 4)
            draw.rectangle(list([box[0],box[1]-10,box[0]+6*len(my_cate[int(label)])+4,box[1]]), outline=color_map(int(label)),width = 4,fill='white')
            draw.text((box[0]+3, box[1]-11), my_cate[int(label)],fill = (0, 0, 0, 100),width = 4)
        plt.imshow(img)
        plt.savefig("./test.jpg")

    return True


if __name__ == "__main__":
    test()
