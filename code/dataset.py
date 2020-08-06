import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import myutils

# IMPORTANT: in the real crack dataset each image was split in many crops and each crop was saved
# In the sealed crack dataset a different strategy is used. The images were saved without any crop
# and the crops are done as "transformations" in the dataloader.
# This helps experementing with different ways of cropping and transforming the images
# For example, we can apply small rotations to the crop (such as +-5 degrees) and resize it to perform data augmentation

def tif2binary(root_dir,file):
    '''
    Input: root directory and file name of a .tif image
    Output: tif image converted to black and white
    '''
    img = Image.open(root_dir + '/B/' + file + '.tif')
    thresh = 128
    fn = lambda x : 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    return r
    
class DatasetRealCrack(Dataset):
    def __init__(self, root, train=True, transform=None):
        Dataset.__init__(self)
        images_dir = root  # os.path.join(root,'images')
        images = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, k) for k in images]

        if train:
            masks_dir = images_dir.replace('/A/', '/B/')  # os.path.join(root,'masks')
            masks = os.listdir(masks_dir)
            self.masks = [os.path.join(masks_dir, k) for k in masks]
            self.masks.sort()
        ######################
        # When uploading the Real Crack dataset to
        # Google drive, it had a consistent problem of
        # downloading some files twice. Let's remove these files.
        # (if there were no duplicate files this piece of code could be deleted)
        # removing duplicate files
        print("len masks: ", len(self.masks))
        print("len images: ", len(self.images))
        im = []
        ma = []
        for i in self.masks:
            ma.append(i[34:]) #the name of the mask files
        for i in self.images:
            im.append(i[34:]) #the name of the image files
        duplicates = list(set(im) - set(ma))
        for duplicate in duplicates:
            self.images.remove(root + duplicate)

        #####################

        self.images.sort()

        print("len masks: ", len(self.masks))
        print("len images: ", len(self.images))
        self.transforms = transform
        self.train = train

    def __getitem__(self, index):
        image_path = self.images[index]

        image = Image.open(image_path).resize([512, 512])
        if self.transforms is not None:
            image = self.transforms(image)
        image = image
        if self.train:
            mask_path = self.masks[index]
            mask = Image.open(mask_path).resize([512, 512])
            if self.transforms is not None:
                mask = self.transforms(mask)
                mask = mask.mean(dim=0).view(1, 512, 512)
                mask[mask < 0] = 0
                mask[mask > 0] = 1

            return image, mask
        return image

    def __len__(self):
        return len(self.images)


class DatasetSealedCrack(Dataset):
    # IMPORTANT: in the real crack dataset each image was split in many crops and each crop was saved
    # In the sealed crack dataset a different strategy is used. The images were saved without any crop
    # and the crops are done as "transformations" in the dataloader.
    # This helps experementing with different ways of cropping and transforming the images
    # For example, we can apply small rotations to the crop (such as +-5 degrees) and resize it to perform data augmentation
    def __init__(self, files, root_dir, transform=None):  # train=True

        """
        Args:
            files (list): list containing the names of the files
            root_dir (string): Directory containing the subdirectories (A and B) with the Cracks and anotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = files
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        img = Image.open(self.root_dir + '/A/' + self.files[idx] + '.png').convert('RGB')
        anotation = tif2binary(self.root_dir, self.files[idx])
        sample = {'image': img, 'anotation': anotation}  # add ids of the images
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['anotation']
    