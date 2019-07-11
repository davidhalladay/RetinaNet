import torch
import torchvision.transforms as transforms

import argparse

from torch.autograd import Variable
import matplotlib.pyplot as plt

import os
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
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

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--ckpt', type=str, default = './checkpoint/ckpt.pth',help='the name of ckpt (default : ./checkpoint/ckpt.pth)')
parser.add_argument('--image_root', type=str, default = "../COCO_dataset/images/train2017",help='root of image data (default : ../COCO_dataset/images/train2017')
parser.add_argument('--pred_root', type=str, default = "./prediction",help='root of predict data (default : ./prediction')
args = parser.parse_args()

#################################################################
#################################################################
def load_checkpoint(checkpoint_path, model,mode='net'):
    state = torch.load(checkpoint_path,map_location = "cuda")
    model.load_state_dict(state[mode])
    print('model loaded from %s' % checkpoint_path)

def visualize(net,device,image_root,image_list,pred_root):
    '''visualize the BBox and cate_pred

    Args:
      image_root: the path of directory where the image data contained
      labels: (tensor) object class labels, sized [#obj,].
      input_size: (int/tuple) model input size of (w,h).

    Returns:
      loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
      cls_targets: (tensor) encoded class labels, sized [#anchors,].
    '''
    for image_name in image_list:
        number = image_name[-10:-4]
        image_path = os.path.join(image_root,image_name)
        # Loading image...
        img = Image.open(image_path)
        w = h = 224
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
            encoder = DataEncoder()
            boxes, labels, scores = encoder.decode(loc_preds.data.cpu().squeeze(), cls_preds.data.cpu().squeeze(), (w,h))
            print("Image number : %s" %(number))
            print("label : ",labels)
            draw = ImageDraw.Draw(img)
            count = 0
            for i,(box,label,score) in enumerate(zip(boxes,labels,scores)):
                count += 1
                draw.rectangle(list(box), outline=color_map(int(label)),width = 2)
                draw.rectangle(list([box[0],box[1]-10,box[0]+6*len(cate[int(label)])+4,box[1]]), outline=color_map(int(label)),width = 2,fill='white')
                draw.text((box[0]+3, box[1]-11), cate[int(label)],fill = (0, 0, 0, 100),width = 2)
                draw.text((box[0]+3, box[1]-10), " "*len(cate[int(label)])+" %.3f"%(score) ,fill = "red",width = 1)
            print("Total number of BBox : ",count)
            img.save(os.path.join(pred_root,"pred_%s.jpg"%(number)))
            print("<"+"="*20+">")

def main():

    print('Loading model from %s'%(args.ckpt))
    net = RetinaNet()
    load_checkpoint('%s'%(args.ckpt),net)
    net.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    visualize(net,device,args.image_root,image_list,args.pred_root)

    return 0


if __name__ == "__main__":
    main()
