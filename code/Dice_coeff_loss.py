import torch
from myutils import onehot

def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    #target[target==0] = -1

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def dice_loss_2_datasets(branch_1, branch_2):
    '''
    Applies the dice loss to predictions of both Real crack and Sealed crack
    Input: 
      - branch_1: dictionary containing real cracks predictions and masks
      - branch_2: dictionary containing sealed cracks predictions and masks
    Output:
      - average dice loss between real and sealed cracks
    '''
    loss_1 = dice_loss(branch_1['outputs'], onehot(branch_1['masks']))
    #print("loss 1:", loss_1)
    loss_2 = dice_loss(branch_2['outputs'], onehot(branch_2['masks']))
    #print("loss 2:", loss_2, "shape: ", branch_2['outputs'].shape, branch_2['masks'].shape)
    return (loss_1 + loss_2) / 2