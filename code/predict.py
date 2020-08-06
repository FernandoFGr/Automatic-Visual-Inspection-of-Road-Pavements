import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import time

from myutils import transform
from model import Unet, Unet_SpatialPyramidPooling, classifier

torch.cuda.manual_seed(777)
torch.manual_seed(777)

img_folder = '../dataset-EdmCrack600-512/A/val/'
img_dir = os.listdir(img_folder)
img_list = [img_folder+k for k in img_dir]
img_list.sort()

unet = Unet(3).cuda()
SC_classifier = classifier(64, 2).cuda()
RC_classifier = classifier(64, 2).cuda()
unet.load_state_dict(torch.load('trained_models/unet_adam_dice_58.pkl'))
SC_classifier.load_state_dict(torch.load('trained_models/SC_classifier_adam_dice_58.pkl'))
RC_classifier.load_state_dict(torch.load('trained_models/RC_classifier_adam_dice_58.pkl'))

total_time = 0
i=0
for file in img_list:
    img = Image.open(file).resize([512,512])
    img = transform(img).cuda().unsqueeze(0)
    start_time = time.time()
    with torch.no_grad():
        pred = unet(img)
        pred = RC_classifier(pred)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    pred = torch.argmin(pred,1)
    pred = pred.squeeze().cpu().numpy()
    pred = 1 - pred
    pred = np.uint8(pred*255)
    pred_img = Image.fromarray(pred)
    img_name = str.split(file, '/')[-1]
    img_name = img_name
    img_name = './test_results/' + img_name
    pred_img.save(img_name, 'png')
    if i%50==0:
        print(i)
    i+=1


print("It takes: {:f}".format(total_time))
