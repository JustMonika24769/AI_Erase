import math
import cv2
import os
import random
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, ToTensor, Resize

def random_horizontal_flip(data):
    if random.random() < 0.5:
        for i, d in enumerate(data):
            data[i] = np.flip(d, axis=(-2))
    return data

def random_crop(data, output_size=(1024, 1024)):
    if random.random() < 0.5:
        h, w = data[0].shape[:2]
        new_h, new_w = output_size
        if h > new_h:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            for i, d in enumerate(data):
                data[i] = d[top:top + new_h, left:left + new_w]
    return data

def square_fill(data, factor=16):
    h, w = data[0].shape[:2]
    X = int(math.ceil(max(h,w)/float(factor))*factor)

    for i, d in enumerate(data):
        blank_fill = np.ones((X, X, 3), dtype=np.uint8)  * 255
        blank_fill[((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = d
        data[i] = blank_fill
    return data

# to genenrate the true mask
def generateMask(src, tar, threshold=10):
    diff_image = paddle.abs(tar - src) * 255
    mean_image = paddle.mean(diff_image, axis=0, keepdim=True)
    mask = paddle.cast((mean_image > threshold), dtype='float32')
    return mask

class ErasingData(Dataset):
    def __init__(self, dataRoot, loadSize, training=True):
        super(ErasingData, self).__init__()
        if training:
            with open(os.path.join(dataRoot, 'train.csv'), 'r') as f:
                self.imageFiles = [l.strip('\n') for l in f.readlines()]
            # self.imageFiles = sorted(self.imageFiles)[:50]
        else:
            with open(os.path.join(dataRoot, 'test.csv'), 'r') as f:
                self.imageFiles = [l.strip('\n') for l in f.readlines()]
        
        self.loadSize = loadSize
        self.ImgTrans = Compose([
                                Resize(size=loadSize, interpolation='bicubic'),
                                ToTensor(), # h * w * c --> c * h * w
                            ])
        self.training = training
    
    def __getitem__(self, index):
        img = cv2.imread(self.imageFiles[index])
        gt = cv2.imread(self.imageFiles[index].replace('images','gts').replace('jpg', 'png'))

        data = [img, gt]
        # make the image to be square (to keep the aspect ratio)
        data = square_fill(data)
        
        # data augment
        if self.training:
            data = random_horizontal_flip(data)
            data = random_crop(data)
        img, gt = data

        img = self.ImgTrans(img)
        gt = self.ImgTrans(gt)

        # mask = 1
        mask = generateMask(img, gt)
        path = self.imageFiles[index].split('/')[-1]
        return {
            'img':img,
            'mask':mask,
            'gt':gt,
            'path':path,
        }
    
    def __len__(self):
        return len(self.imageFiles)

if __name__ == '__main__':
    paddle.set_device('gpu:1')
    dataset = ErasingData('Dataset/Baidu_Dewords/dehw_train_dataset', (256, ) * 2, training=True)
    a = dataset[1]
