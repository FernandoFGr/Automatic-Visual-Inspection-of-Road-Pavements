import numpy as np
import cv2
import os

def combine(in_dir, out_dir, key):

    file_list = os.listdir(in_dir)
    file_list.sort()

    os.makedirs(out_dir, exist_ok=True)
    output_dict = dict()
    width_max_dict = dict()
    height_max_dict = dict()
    fill_dict =dict()

    for file in file_list:
        print(file)
        if key in file:
            id, width_idx, height_idx = file.split('.')[0].split('_')[0].split('--')
            width_idx = int(width_idx)
            height_idx = int(height_idx)

            if id not in output_dict:
                output_dict[id] = np.zeros((1080, 1920))
                fill_dict[id] = 0

            output = cv2.imread(in_dir+'/'+file,cv2.IMREAD_GRAYSCALE) >= 128
            target_height, target_width = output.shape

            output_dict[id][height_idx:height_idx + target_height,width_idx:width_idx + target_width] += output
            fill_dict[id] += 1

            if id not in width_max_dict:
                width_max_dict[id] = width_idx + target_width
            else:
                if width_idx + target_width > width_max_dict[id]:
                    width_max_dict[id] = width_idx + target_width

            if id not in height_max_dict:
                height_max_dict[id] = height_idx + target_height
            else:
                if height_idx + target_height > height_max_dict[id]:
                    height_max_dict[id] = height_idx + target_height

            if fill_dict[id] == 28:
                cv2.imwrite(out_dir + '/' + id + '.png', output_dict[id] * 255)
                fill_dict.pop(id, None)
                output_dict.pop(id, None)
                width_max_dict.pop(id, None)
                height_max_dict.pop(id, None)


if __name__ == "__main__":
    combine('./test_results', './combined/raw_prediction', '')
    combine('../dataset-EdmCrack600-512/B/test', './combined/ground truth', '')