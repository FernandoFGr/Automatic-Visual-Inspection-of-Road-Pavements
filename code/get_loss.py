import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from myutils import transform, cal_iou, onehot, data_Train_transforms, data_Test_transforms
import myutils
from model import Unet, classifier, Unet_SpatialPyramidPooling
from dataset import DatasetSealedCrack, DatasetRealCrack
from Dice_coeff_loss import dice_loss, dice_loss_2_datasets
from train import ConcatDataset


def get_loss_from_loader(data_loader, unet, SC_classifier, RC_classifier):
    '''
    -Calculates the average dice loss between both datasets.
    -As the sealed crack dataset has only ~300 images (they were not cropped), it starts to repeat. 
    Each time it repeats new random crops are taken.
    -As the real crack images were cropped before being saved (more that 10k smaller images),
    Each image is seen only once (no repetition)
    '''
    totalloss = 0
    for i, (realCrack_batch, sealedCrack_batch) in enumerate(data_loader):
        SC_images = sealedCrack_batch[0].cuda()
        SC_masks = sealedCrack_batch[1].cuda()
        RC_images = realCrack_batch[0].cuda()
        RC_masks = realCrack_batch[1].cuda()
        with torch.no_grad():
            SC_encoder = unet(SC_images)
            RC_encoder = unet(RC_images)
            SC_outputs = SC_classifier(SC_encoder)
            RC_outputs = RC_classifier(RC_encoder)
        branch_RC = {'outputs': RC_outputs, 'masks': RC_masks}
        branch_SC = {'outputs': SC_outputs, 'masks': SC_masks}
        loss = dice_loss_2_datasets(branch_RC, branch_SC)
        totalloss += loss * RC_images.size(0)  # *2?
        if i == 3400: #limits the amount of time calculating the loss
            return totalloss / (i+1)
    return totalloss / len(data_loader)
##########################
# EXAMPLE of code to run the function "get_loss_from_loader": 
#SC_root_dir = '../dataset-EdmSealedCrack-512'
#train_files, val_files, test_files = myutils.organize_SC_files(SC_root_dir)

#train_RC_dataset = DatasetRealCrack('../dataset-EdmCrack600-512/A/train', transform=transform)
#train_SC_dataset = DatasetSealedCrack(files=train_files, root_dir=SC_root_dir, transform=data_Train_transforms)
#val_RC_dataset = DatasetRealCrack('../dataset-EdmCrack600-512/A/test', transform=transform)
#val_SC_dataset = DatasetSealedCrack(files=val_files, root_dir=SC_root_dir, transform=data_Train_transforms)
#test_RC_dataset = DatasetRealCrack('../dataset-EdmCrack600-512/A/val', transform=transform)
#test_SC_dataset = DatasetSealedCrack(files=test_files, root_dir=SC_root_dir, transform=data_Train_transforms)
#test_loader = torch.utils.data.DataLoader(
#    ConcatDataset(
#        test_RC_dataset,
#        test_SC_dataset
#    ),
#    batch_size=1, shuffle=False,
#    num_workers=2)    
#get_loss_from_loader(test_loader, unet, SC_classifier, RC_classifier)
############################
if __name__ == '__main__':
    get_loss_models()