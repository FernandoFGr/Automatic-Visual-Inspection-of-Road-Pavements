import ignite
import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from focalloss import FocalLoss

def F1_score_across_dataloader(data_loader, model, classifier):
    '''
    calculates the average F1 score, precision and recall across a dataloader
    '''
    TP = 0
    FP = 0
    FN = 0
    sum_precisions = 0
    sum_recalls = 0
    sum_f1 = 0
    i=0
    for image, anotation in data_loader:
        image = image.to('cuda')
        anotation = anotation.type('torch.LongTensor')
        anotation = anotation.to('cuda')
        with torch.no_grad():
            outputs = model(image)
            outputs = classifier(outputs)

        preds = outputs.argmax(1)
        #preds = outputs[:,1,:,:] > 0.3
        tp = torch.sum(preds*anotation)
        fp = torch.sum(preds*((anotation+1)%2))
        fn = torch.sum(anotation*((preds+1)%2))

        TP += tp
        FP += fp
        FN += fn 
        p = TP/(TP + FP + 1e-20)
        r = TP/(TP + FN + 1e-20)
        F1 = 2* (p * r)/(p + r + 1e-20)
        if torch.sum(anotation) != 0:
            i+=1
            p_single_img = tp/(tp + fp + 1e-20)
            r_single_img = tp/(tp + fn + 1e-20)
            f1_single_img = 2* (p_single_img * r_single_img)/(p_single_img + r_single_img + 1e-20)
            sum_precisions += p_single_img
            sum_recalls += r_single_img
            sum_f1 += f1_single_img
            
        if ((i+1)%10) == 0:
            print(TP, FP, FN)
            print("average precision: ", sum_precisions/i)
            print("average recall: ", sum_recalls/i)
            print("average f1: ", sum_f1/i)
            print("\n")
            print("f1 calculated with avg precision and recall: ", 2* ((sum_precisions/i) * (sum_recalls/i))/((sum_precisions/i) + (sum_recalls/i) + 1e-20))
            print("TOTAL PRECISION: ", p)
            print("TOTAL RECALL: ", r)
            print("TOTAL F1: ", F1)
            print("\n")
    print("final: ")
    print("(final) TOTAL PRECISION: ", p)
    print("(final) TOTAL RECALL: ", r)
    print("(final) TOTAL F1: ", F1)
    print("average precision: ", sum_precisions/i)
    print("average recall: ", sum_recalls/i)
    print("average f1: ", sum_f1/i)


def pixel_accuracy(data_loader, model, classifier):
    sum = 0
    i=0
    positive = 0
    total = 0
    for image, anotation in data_loader:
        image = image.to('cuda')
        anotation = anotation.type('torch.LongTensor')
        anotation = anotation.to('cuda')
        with torch.no_grad():
            outputs = model(image)
            outputs = classifier(outputs)

        preds = outputs.argmax(1) 

        positive += torch.sum(anotation == preds)
        num_pixels_per_img = preds.shape[1]*preds.shape[2]
        #total += 512*512
        total += num_pixels_per_img
        print(positive/(total + 1e-20))
    print("final: ")
    print(positive/(total + 1e-20))
    return positive/(total + 1e-20)




