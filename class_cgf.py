import numpy as np
from PIL import ImageColor

cate = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane',
        5:'bus', 6:'train', 7:'truck', 8:'boat', 9:'traffic light',
        10:'fire hydrant', 12:'stop sign', 13:'parking meter', 14:'bench',
        15:'bird' ,16:'cat',17:'dog',18:'horse',19:'sheep',20:'cow',21:'elephant',
        22:'bear',23:'zebra',24:'giraffe',26:'backpack',27:'umbrella',30:'handbag',
        31:'tie',32:'suitcase',33:'frisbee',34:'skis',35:'snowboard',36:'sports ball',
        37:'kite',38:'baseball bat',39:'baseball glove',40:'skateboard',41:'surfboard',
        42:'tennis racket',43:'bottle',45:'wine glass',46:'cup',47:'fork',
        48:'knife',49:'spoon',50:'bowl',
        51:'banana',52:'apple',53:'sandwich',54:'orange',55:'broccoli',
        56:'carrot',57:'hot dog',58:'pizza',59:'donut',60:'cake',61:'chair',
        62:'couch',63:'potted plant',64:'bed',66:'dining table',69:'toilet',
        71:'tv',72:'laptop',73:'mouse',74:'remote',75:'keyboard',76:'cell phone',
        77:'microwave',78:'oven',79:'toaster',80:'sink',81:'refrigerator',83:'book',
        84:'clock',85:'vase',86:'scissors',87:'teddy bear',88:'hair drier',89:'toothbrush'}

my_cate = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane',
        5:'bus', 6:'train', 7:'truck', 8:'boat', 9:'traffic light',
        10:'fire hydrant', 11:'stop sign', 12:'parking meter', 13:'bench',
        14:'bird' ,15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
        21:'bear',22:'zebra',23:'giraffe',24:'backpack',
        25:'umbrella',26:'handbag',
        27:'tie',28:'suitcase',29:'frisbee',30:'skis',31:'snowboard',32:'sports ball',
        33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',
        38:'tennis racket',39:'bottle',40:'wine glass',41:'cup',42:'fork',
        43:'knife',44:'spoon',45:'bowl',
        46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',
        57:'couch',58:'potted plant',59:'bed',60:'dining table',61:'toilet',
        62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',
        68:'microwave',69:'oven',70:'toaster',71:'sink',72:'refrigerator',73:'book',
        74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush'}

coco_list = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,
            28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,
            54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,
            82,84,85,86,87,88,89,90]

my_list = [i for i in range(0,80)]

def class_map(number,mode = 'coco2my'):
    if mode == 'coco2my':
        return my_list[coco_list.index(number)]
    if mode == 'my2coco':
        return coco_list[my_list.index(number)]
    else:
        Print("ERROR: Wrong input!")
        return 9999999

def color_map(number):
    return list(ImageColor.colormap.items())[number][0]




#######################################################
#######################################################
#######################################################
#######################################################

coco_cate = [
        {'supercategory': 'person', 'id': 1, 'name': 'person'},
        {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
        {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
        {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
        {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
        {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
        {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
        {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
        {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
        {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
        {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
        {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
        {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
        {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
        {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
        {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
        {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
        {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
        {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
        {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
        {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
        {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
        {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
        {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
        {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
        {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
        {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
        {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
        {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
        {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
        {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
        {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
        {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
        {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
        {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
        {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
        {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
        {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
        {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
        {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
        {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
        {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
        {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
        {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
        {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
        {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
        {'supercategory': 'food', 'id': 52, 'name': 'banana'},
        {'supercategory': 'food', 'id': 53, 'name': 'apple'},
        {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
        {'supercategory': 'food', 'id': 55, 'name': 'orange'},
        {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
        {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
        {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
        {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
        {'supercategory': 'food', 'id': 60, 'name': 'donut'},
        {'supercategory': 'food', 'id': 61, 'name': 'cake'},
        {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
        {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
        {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
        {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
        {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
        {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
        {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
        {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
        {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
        {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
        {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
        {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
        {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
        {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
        {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
        {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
        {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
        {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
        {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
        {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
        {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
        {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
        {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
        {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}
        ]
