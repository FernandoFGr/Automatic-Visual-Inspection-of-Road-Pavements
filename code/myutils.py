import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import os
from torchvision import transforms, utils
from focalloss import FocalLoss

mean=[.5,.5,0.5]
std=[.5,.5,0.5]
###############
#Deep lab v3
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]

def cal_iou(model, dataset):
    pa=mpa=miou=fwiou=0.
    for img, mask in dataset:
        mask = mask.cuda()
        with torch.no_grad():
            pred = model(img.unsqueeze(0).cuda())
            pred = torch.argmax(pred, 1).float()
        pa += get_pa(pred, mask)
        mpa += get_mpa(pred, mask)
        miou += get_miou(pred, mask)
        fwiou += get_fwiou(pred, mask)
    lenth = len(dataset)
    pa /= lenth
    mpa /= lenth
    miou /= lenth
    fwiou /= lenth
    return pa.item(), mpa.item(), miou.item(), fwiou.item()

def get_pa(pred, mask):
    return (pred==mask).sum().float()/(512*512)


def get_mpa(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum().float()/\
            (mask_crack.sum())/2 +\
            (pred_fine*mask_fine).sum().float()/\
            (mask_fine.sum())/2


def get_miou(pred, mask):

    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum().float()/\
            ((mask_crack+pred_crack)!=0).sum().float()/2+\
            (pred_fine*mask_fine).sum().float()/\
            ((mask_fine+pred_fine)!=0).sum().float()/2


def get_fwiou(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return  mask_crack.sum()*(pred_crack*mask_crack).sum().float()/\
            ((mask_crack+pred_crack)!=0).sum().float()/(512*512)+\
            mask_fine.sum()*(pred_fine*mask_fine).sum().float()/\
            ((mask_fine+pred_fine)!=0).sum().float()/(512*512)


def onehot(masks):
    masks_t = torch.zeros(masks.size(0), 2, 
                    masks.size(2), masks.size(3)).cuda()
    masks_t[:,0,:,:][masks[:,0,:,:]==0] = 1
    masks_t[:,1,:,:][masks[:,0,:,:]==1] = 1   
    return masks_t

class ToTensor():
    def __call__(self, sample):
        '''
        Input:
          - sample: dictionary containing PIL image and PIL anotation
        Output:
          - dictionary containing the images converted to torch tensors
        '''
        image, anotation = sample['image'], sample['anotation']
        image = transforms.ToTensor()(image)
        anotation = transforms.ToTensor()(anotation)
        return {'image': image, 'anotation': anotation}

class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        '''
        Input:
          - sample: dictionary containing PIL image and PIL anotation
        Output:
          - dictionary containing the normalize input images and the annotations unchanged
        '''
        image, anotation = sample['image'], sample['anotation']
        image = transforms.Normalize(mean=self.mean, std=self.std)(image)
        return {'image': image, 'anotation': anotation}

class RandomResizedCrop():
    def __init__(self, height, width):
        self.width = width
        self.height = height
    def __call__(self, sample):
        image, annotation = sample['image'], sample['anotation']
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.height, self.width))
        image = transforms.functional.crop(image, i, j, h, w)
        annotation = transforms.functional.crop(annotation, i, j, h, w)
        return {'image': image, 'anotation': annotation}

class ResizeImage():
    def __init__(self, height, width, min, max):
        self.width = width
        self.height = height
        self.height = height
        self.min = min
        self.max = max
    def __call__(self, sample):
        resize_factor = self.min + (random.random() * (self.max - self.min))
        image, annotation = sample['image'], sample['anotation']
        resize = transforms.Resize(size=(int(self.height * resize_factor), int(self.width * resize_factor)))
        image = resize(image)
        annotation = resize(annotation)
        return {'image': image, 'anotation': annotation}

class RandomRotation():
    def __init__(self, max):
        self.max = max
    def __call__(self, sample):
        rotation_value = -self.max + (random.random() * (self.max + self.max))
        image, annotation = sample['image'], sample['anotation']
        rotation = transforms.RandomRotation((rotation_value,rotation_value))
        image = rotation(image)
        annotation = rotation(annotation.convert('RGB'))
        annotation = annotation.convert('1')
        return {'image': image, 'anotation': annotation}

class RandomHorizontalFlip():
    def __call__(self, sample):
        image, annotation = sample['image'], sample['anotation']
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            annotation = transforms.functional.hflip(annotation)
        return {'image': image, 'anotation': annotation}

def organize_SC_files(root_dir):
    # ORGANIZING THE SEALED CRACK DATASETS
    # creating lists containing the names of the images belonging to train/val/test set 
    files = os.listdir(root_dir + '/B/') # B: annotations
    #files.remove('desktop.ini') # extra file on the dataset
    files.remove('test36-1024.tif') # wrong shape
    files.remove('test36-1081.tif')
    files.remove('test37-474.tif') # wrong background image
    files.sort()
    files = [f[:-4] for f in files]
    train_files=[]
    val_files=[]
    test_files=[]
    for i in range(len(files)):
        if i%10 < 7:
            train_files.append(files[i])
        elif i%10 < 9:
            val_files.append(files[i])
        else:
            test_files.append(files[i])

    print(len(train_files), len(val_files),len(test_files))
    return train_files, val_files, test_files

transform=transforms.Compose([
    #RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean,std=std) 
])


# Below are the transformations to apply to the sealed crack dataset (not to the real cracks, as the images are saved after being cropped).
# data_Train_transforms should be applied during training: performs data augmentation and crops size 512x512
# data_Test_transforms should be applied during testing: no data augmentation; image as large as the model supports 1024x1792
data_Train_transforms = transforms.Compose([RandomRotation(5),
                                      ResizeImage(1080,1920, 0.8, 1.2), 
                                      RandomResizedCrop(512,512), #change
                                      RandomHorizontalFlip(),
                                      ToTensor(), 
                                      Normalize(mean=mean, std=std)])

data_Test_transforms = transforms.Compose([RandomResizedCrop(1024,1792),#RandomResizedCrop(1072,1920),
                                      #RandomHorizontalFlip(),
                                      ToTensor(), 
                                      Normalize(mean=mean, std=std)])