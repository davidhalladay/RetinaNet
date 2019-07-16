'''Predict the image and label txt file for mAP calculating.

This program will auto create the prediction file.
Please using 'python3 prediction.py -h' first to realize the usage of args.

Author :    DavidFan in Irvine, CA, USA

'''
import torch
import torchvision.transforms as transforms

import argparse
import random
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont
from class_cgf import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

image_list = ['000000524387.jpg','000000229494.jpg','000000162998.jpg',
            '000000392035.jpg','000000533312.jpg','000000454959.jpg','000000022240.jpg',
            '000000201918.jpg','000000456376.jpg','000000124122.jpg','000000033802.jpg',
            '000000020972.jpg','000000123532.jpg','000000575837.jpg','000000574028.jpg',
            '000000251723.jpg','000000312718.jpg','000000213753.jpg','000000021138.jpg']

val_image_list = ['000000001353.jpg','000000001296.jpg','000000020571.jpg','000000020553.jpg','000000006460.jpg','000000004795.jpg','000000001490.jpg','000000002153.jpg','000000002587.jpg','000000025181.jpg',
                '000000316404.jpg','000000218439.jpg','000000314177.jpg','000000468505.jpg']

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--ckpt', type=str, default = './checkpoint/ckpt_alt.pth',help='the name of ckpt (default : ./checkpoint/ckpt_alt.pth)')
parser.add_argument('--image_root', type=str, default = "../COCO_dataset/images/val2017",help='root of image data (default : ../COCO_dataset/images/val2017')
parser.add_argument('--anno_root', type=str, default = "./data/val2017.txt",help='root of annotation (default : ./data/val2017.txt')
parser.add_argument('--pred_root', type=str, default = "./prediction",help='root of predict data (default : ./prediction')
parser.add_argument('--count_limit','-c', action='store_true',help='prediction limit')
parser.add_argument('--vmode','-v', action='store_true', help='mode for visualize')
parser.add_argument('--mmode','-m', action='store_true', help='mode for output mPA labels. Note: for mPA mode you should check the root path to image.')
args = parser.parse_args()

#################################################################
#################################################################
def load_checkpoint(checkpoint_path, model,mode='net'):
    state = torch.load(checkpoint_path,map_location = "cuda")
    model.load_state_dict(state[mode])
    print('model loaded from %s' % checkpoint_path)

def visualize(encoder,net,device,image_root,image_list,pred_root):
    '''visualize the BBox and cate_pred

    Args:
      image_root: the path of directory where the image data contained
      labels: (tensor) object class labels, sized [#obj,].
      input_size: (int/tuple) model input size of (w,h).

    Note:
      loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
      cls_targets: (tensor) encoded class labels, sized [#anchors,].
    '''
    print("<"+"="*20+">")
    print("[Processing visualization]")
    for i, image_name in enumerate(image_list):
        number = image_name[-10:-4]
        image_path = os.path.join(image_root,image_name)
        # Loading image...
        img = Image.open(image_path)
        w = h = 360
        img = img.resize((w,h))

        x = transform(img)
        x = x.unsqueeze(0)
        if torch.cuda.is_available():
            net = net.to(device)
            x = x.to(device)

        with torch.no_grad():
            x = Variable(x)
            loc_preds, cls_preds = net(x)

            # Decoding...
            print("location preds (shape): ",loc_preds.shape)
            print("class preds (shape): ",cls_preds.shape)
            boxes, labels, scores = encoder.decode(loc_preds.data.cpu().squeeze(), cls_preds.data.cpu().squeeze(), (w,h))
            print("Image number : %s" %(number))
            print("Output label : ",labels)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("./font/DELIA_regular.ttf", 20)
            count = 0
            for i,(box,label,score) in enumerate(zip(boxes,labels,scores)):
                count += 1
                draw.rectangle(list(box), outline=color_map(int(label)),width = 3)
                draw.rectangle(list([box[0],box[1]-18,box[0]+10*len(my_cate[int(label)])+5,box[1]]), outline=color_map(int(label)),width = 2,fill='white')
                draw.text((box[0]+3, box[1]-16), my_cate[int(label)] + ": %.3f"%(score.float()),font = font,fill = (0, 0, 0, 100),width = 4)
        img.save(os.path.join(pred_root,"pred_%s.jpg"%(number)))
        print("<"+"="*20+">")
    print("Total number of BBox : ",count)
    print("[Done]")
    print("<"+"="*20+">")
    return True

def mPA_pred(encoder,net,device,image_root,anno_root):
    '''pred labels for mPA
    Args:
      image_root: the path of directory where the image data contained
      labels: (tensor) object class labels, sized [#obj,].
      input_size: (int/tuple) model input size of (w,h).

    Note:
      loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
      cls_targets: (tensor) encoded class labels, sized [#anchors,].
    '''
    print("<"+"="*20+">")
    print("[Processing labels prediction for mPA]")

    with open(anno_root) as f:
        lines = f.readlines()
    fnames = []
    for line in lines:
        splited = line.strip().split()
        fnames.append(splited[0])

    print("Total number of image : ",len(fnames))
    w = h = 360
    print("Using image size = (%d,%d)"%(w,h))

    count = 0
    comb_tmp = []
    sample = random.sample([i for i in range(len(fnames))], 100)
    if args.count_limit :
        fnames = np.array(fnames)[sample]
        print(fnames)
    for i, image_name in enumerate(tqdm(fnames)):
        number = image_name[-10:-4]
        image_path = os.path.join(image_root,image_name)
        # Loading image...
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((w,h))

        x = transform(img)
        x = x.unsqueeze(0)

        x = x.to(device)

        with torch.no_grad():
            x = Variable(x)
            loc_preds, cls_preds = net(x)

            # Decoding...
            boxes, labels, scores = encoder.decode(loc_preds.data.cpu().squeeze(), cls_preds.data.cpu().squeeze(), (w,h))

            file = open("./mPA/detection-results/%s.txt"%(image_name.replace(".jpg","")),"w")
            for i,(bbox,label,score) in enumerate(zip(boxes,labels,scores)):
                file.write("%s %.3f %.3f %.3f %.3f %.3f\n"%(my_cate[int(label)],float(score),float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])))
        count += 1


    print("Total number of BBox : ",count)
    print("[Done]")
    print("<"+"="*20+">")

def main():

    print('Loading model from %s'%(args.ckpt))
    net = RetinaNet()
    load_checkpoint('%s'%(args.ckpt),net)
    net.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    net = net.to(device)

    encoder = DataEncoder()

    if args.vmode :
        if not os.path.exists("%s"%(args.pred_root)):
            os.makedirs("%s"%(args.pred_root))
        visualize(encoder,net,device,args.image_root,val_image_list,args.pred_root)
    if args.mmode :
        if not os.path.exists("./mPA"):
            os.makedirs("./mPA")
        if os.path.exists("./mPA/detection-results"):
            print("Remove Pred file")
            os.system("rm -rf ./mPA/detection-results")
            os.makedirs("./mPA/detection-results")
        if not os.path.exists("./mPA/detection-results"):
            os.makedirs("./mPA/detection-results")
        mPA_pred(encoder,net,device,args.image_root,args.anno_root)

    return 0


if __name__ == "__main__":
    main()
