'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random
import cv2
from retinanet import RetinaNet
from torch.autograd import Variable
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop

from pycocotools.coco import COCO
from class_cgf import *

class ListDataset(data.Dataset):
    def __init__(self, root, ann_root, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.ann_root = ann_root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = os.listdir(self.root)
        self.fnames.sort()
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()
        self.coco = COCO(self.ann_root)

        self.len = len(self.fnames)
        print("<"+"=====Initializaion succeed====="+">")

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img_num = int(fname.replace(".jpg",""))
        img = Image.open(os.path.join(self.root,fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        annIds = self.coco.getAnnIds(imgIds=[img_num], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        boxes = []
        labels = []
        for i, ann in enumerate(anns):
            coco_label = int(ann['category_id'])
            label = class_map(coco_label)
            boxes.append(ann['bbox'])
            labels.append(label)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []

        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.len


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataType = 'train2017'
    root_path = "../COCO_dataset/images/%s"%(dataType)
    ann_root_path = "../COCO_dataset/annotations/instances_%s.json"%(dataType)
    dataset = ListDataset(root=root_path,ann_root=ann_root_path, train=True, transform=transform, input_size=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        print(images.shape)
        print(loc_targets.shape)
        print(cls_targets.shape)
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')


        print('Loading image..')
        net = RetinaNet()
        net.eval()

        img = Image.open('a.jpg')
        w = h = 224
        img = img.resize((w,h))

        print('Predicting..')
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('Device used:', device)
        if torch.cuda.is_available():
            net = net.to(device)
            x = x.to(device)
            loc_targets = loc_targets.to(device)

        with torch.no_grad():
            loc_preds, cls_preds = net(x)
            print(cls_preds.shape)
            print('Decoding..')
            encoder = DataEncoder()
            boxes, labels = encoder.decode(loc_targets.data[0].cpu(), cls_preds.data.cpu().squeeze(), (w,h))
            print('class : ',cls_targets)
            draw = ImageDraw.Draw(img)
            for i,(box,label) in enumerate(zip(boxes,labels)):
                draw.rectangle(list(box), outline='green')
                draw.text((box[0], box[1]), str(label + 1))
            plt.imshow(img)
            plt.savefig("./test.png")

        break

if __name__ == "__main__":
    test()
