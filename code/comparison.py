import numpy as np
from scipy.ndimage.measurements import label
import cv2
import os

def comparison(gt_dir, pred_dir):
    file_list = os.listdir(pred_dir)

    P_total = 0
    R_total = 0
    F1_total = 0

    num = 0
    for file in file_list:
        print(file)
        pred = np.bool_(cv2.imread(pred_dir+'/'+file,cv2.IMREAD_GRAYSCALE) >= 128)
        gt = np.bool_(cv2.imread(gt_dir+'/'+file,cv2.IMREAD_GRAYSCALE) >= 128)



        padded_gt = np.pad(gt, ((2, 2), (2, 2)), 'constant', constant_values=((0,0),(0,0)))
        #padded_pred = np.pad(pred, ((2, 2), (2, 2)), 'constant', constant_values=((0,0),(0,0)))
        #
        FN = np.sum(~pred * gt)



        #
        gt_area = padded_gt.copy()
        for i in range(-2,3):
            for j in range(-2,3):
                gt_area = np.logical_or(gt_area, np.roll(np.roll(padded_gt, i, axis=0), j, axis=1))

        TP = np.sum(pred * gt_area[2:-2,2:-2])
        FP = np.sum(pred * ~gt_area[2:-2,2:-2])

        if FP==0:
            P = 1
        else:
            P = TP/(TP+FP)
        if FN==0:
            R = 1
        else:
            R = TP/(TP+FN)
        if P+R==0:
            F1 = 0
        else:
            F1 = 2*P*R/(P+R)
        P_total += P
        R_total += R
        F1_total += F1
        print(file,P,R, F1)
        num += 1
        print('Average now')
        print(P_total/num,R_total/num, F1_total/num)
    P_average = P_total/num
    R_average = R_total/num
    F1_average = F1_total/num
    print('Average P, R, F1, Average F1:')
    print(P_average, R_average, F1_average, 2*P_average*R_average/(P_average+R_average))
    return P_average, R_average, F1_average

if __name__ == "__main__":
    comparison('./combined/ground truth', './combined/raw_prediction')