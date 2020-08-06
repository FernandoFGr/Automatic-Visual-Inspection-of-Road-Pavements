import logging
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from myutils import transform, cal_iou, onehot, data_Train_transforms, data_Test_transforms
import myutils
from model import Unet, classifier, Unet_SpatialPyramidPooling
from dataset import DatasetSealedCrack, DatasetRealCrack

import lovasz_losses as L
from Dice_coeff_loss import dice_loss, dice_loss_2_datasets
from focalloss import FocalLoss
import focalloss
from multiprocessing import freeze_support
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import segmentation_models_pytorch as smp

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)


def unet_train():

    batch_size = 1
    num_epochs = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    num_workers = 2
    lr = 0.0001

    losslist = ['dice']  # ['focal', 'bce', 'dice', 'lovasz']
    optimlist = ['adam']  # ['adam', 'sgd']
    iflog = True

    SC_root_dir = '../dataset-EdmSealedCrack-512'
    train_files, val_files, test_files =  myutils.organize_SC_files(SC_root_dir)
 
    train_RC_dataset = DatasetRealCrack('../dataset-EdmCrack600-512/A/train', transform=transform)
    train_SC_dataset = DatasetSealedCrack(files=train_files, root_dir= SC_root_dir, transform=data_Train_transforms)
    val_RC_dataset = DatasetRealCrack('../dataset-EdmCrack600-512/A/val', transform=transform)
    val_SC_dataset = DatasetSealedCrack(files=val_files, root_dir= SC_root_dir, transform=data_Test_transforms)


    train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 train_RC_dataset,
                 train_SC_dataset
             ),
             batch_size=2, shuffle=True,
             num_workers=2)

    criterion = nn.BCELoss()
    focallos = FocalLoss(gamma=2)
    doubleFocalloss = focalloss.FocalLoss_2_datasets(gamma=2)

    epoidx = -1
    for los in losslist:
        for opt in optimlist:
            start = time.time()
            print(los, opt)
            torch.manual_seed(77)
            torch.cuda.manual_seed(77)
            #################
            #unet = Unet_SpatialPyramidPooling(3).cuda()
            #################
            unet = Unet(3).cuda()
            SC_classifier = classifier(64, 2).cuda()
            RC_classifier = classifier(64, 2).cuda()

            ##################
            #unet = smp.Unet('resnet34', encoder_weights='imagenet').cuda()
            #unet.segmentation_head = torch.nn.Sequential().cuda()
            #SC_classifier = classifier(16, 2).cuda()
            #RC_classifier = classifier(16, 2).cuda()


            #UNCOMMENT TO KEEP TRAINING THE BEST MODEL
            prev_epoch = 0 # if loading model 58, change to prev_epoch = 58. When saving the model, it is going to be named as 59, 60, 61...
            #unet.load_state_dict(torch.load('trained_models/unet_adam_dice_58.pkl'))
            #SC_classifier.load_state_dict(torch.load('trained_models/SC_classifier_adam_dice_58.pkl'))
            #RC_classifier.load_state_dict(torch.load('trained_models/RC_classifier_adam_dice_58.pkl'))

            history = []
            if 'adam' in opt:
                optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
            elif 'sgd' in opt:
                optimizer = torch.optim.SGD(unet.parameters(), lr=10 * lr, momentum=0.9)

            logging.basicConfig(filename='./logs/logger_unet.log', level=logging.INFO)

            total_step = len(train_loader)
            epoidx += 1
            for epoch in range(num_epochs[epoidx]):
                totalloss = 0
                for i, (realCrack_batch, sealedCrack_batch) in enumerate(train_loader):
                    SC_images = sealedCrack_batch[0].cuda()
                    SC_masks = sealedCrack_batch[1].cuda()
                    RC_images = realCrack_batch[0].cuda()
                    RC_masks = realCrack_batch[1].cuda()
                    SC_encoder = unet(SC_images)
                    RC_encoder = unet(RC_images)
                    #############
                    SC_outputs = SC_classifier(SC_encoder)
                    RC_outputs = RC_classifier(RC_encoder)
                    #############
                    #Deep lab v3
                    #SC_outputs = SC_classifier(SC_encoder['out'])
                    #RC_outputs = RC_classifier(RC_encoder['out'])
                    ##############
                    if 'bce' in los:
                        masks = onehot(masks)
                        loss = criterion(outputs, masks)
                    elif 'dice' in los:
                        branch_RC = {'outputs': RC_outputs,'masks': RC_masks}
                        branch_SC = {'outputs': SC_outputs,'masks': SC_masks}
                        loss = dice_loss_2_datasets(branch_RC, branch_SC)
                        #masks = onehot(masks)
                        #loss = dice_loss(outputs, masks)
                    elif 'lovasz' in los:
                        masks = onehot(masks)
                        loss = L.lovasz_hinge(outputs, masks)
                    elif 'focal' in los:
                        #loss = focallos(outputs, masks.long())
                        branch_RC = {'outputs': RC_outputs,'masks': RC_masks.long()}
                        branch_SC = {'outputs': SC_outputs,'masks': SC_masks.long()}
                        loss = doubleFocalloss(branch_RC, branch_SC)
                    totalloss += loss * RC_images.size(0) #*2?
                    #print(RC_images.size(0))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % 10 == 0:
                        print(epoch, i)
                        print("total loss: ", totalloss)
                    if i % 1000 == 0:
                        print("Epoch:%d;     Iteration:%d;      Loss:%f" % (epoch, i, loss))
                
                    if i + 1 == total_step:  # and epoch%1==0: #and val_miou>0.85:
                        torch.save(unet.state_dict(),
                                    './trained_models/unet_' + opt + '_' + los + '_' + str(epoch + 1 + prev_epoch) + '.pkl')
                        torch.save(RC_classifier.state_dict(),
                                    './trained_models/RC_classifier_' + opt + '_' + los + '_' + str(epoch + 1 + prev_epoch) + '.pkl')
                        torch.save(SC_classifier.state_dict(),
                                    './trained_models/SC_classifier_' + opt + '_' + los + '_' + str(epoch + 1 + prev_epoch) + '.pkl')
                history_np = np.array(history)
                np.save('./logs/unet_' + opt + '_' + los + '.npy', history_np)
            end = time.time()
            print((end - start) / 60)

if __name__ == '__main__':
    freeze_support()
    unet_train()
        
        