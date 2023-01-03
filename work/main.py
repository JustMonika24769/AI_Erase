import math
import os
import argparse
import numpy as np
import tqdm
import copy
import time
import random
import visdom

import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
import paddle.nn.functional as F

from dataloader import *
from utils import *
from model import *
from loss import *

MODEL = {
    'segformer_b1': segformer_b1,
    'segformer_b2': segformer_b2,
}


class Session:
    def __init__(self, config=None):
        self.config = config
        paddle.set_device('gpu:0')  # can modify the gpu you want to use

    def init_env(self):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)

        if self.config.show_visdom:
            self.plotter = visdom.Visdom(env='main', port=7000)

        os.makedirs(os.path.join(self.config.modelsSavePath, self.config.arch), exist_ok=True)
        self.timestamp = time.strftime('%m%d%H%M', time.localtime(time.time()))

        self.load_data()
        self.load_model()

    def load_data(self):
        batchSize = self.config.batchSize
        loadSize = (self.config.loadSize,) * 2
        dataRoot = self.config.dataRoot
        numOfWorkers = 0#self.config.numOfWorkers

        trainData = ErasingData(dataRoot, loadSize, training=True)
        self.trainData = DataLoader(trainData, batch_size=batchSize,
                                    shuffle=True, num_workers=numOfWorkers, drop_last=True)

        evalData = ErasingData(dataRoot, loadSize, training=False)
        self.evalData = DataLoader(evalData, batch_size=batchSize,
                                   shuffle=False, num_workers=numOfWorkers, drop_last=False)

    def load_model(self):
        net = MODEL[self.config.arch](num_classes=1, pretrained=self.config.pretrained)
        criterion = MaskLoss()

        params = np.sum([p.numel() for p in net.parameters()]).item() * 4 / 1024 / 1024
        print(f"Loaded Enhance Net parameters : {params:.3e} MB")

        if self.config.parallel:
            numOfGPUs = paddle.device.cuda.device_count()
            if numOfGPUs > 1:
                net = paddle.DataParallel(net, device_ids=range(numOfGPUs))
                criterion = paddle.DataParallel(criterion, device_ids=range(numOfGPUs))
            else:
                self.config.parallel = False

        self.net = net
        self.criterion = criterion

        ds = int(self.config.num_epochs * 1000 / self.config.batchSize)
        self.scheduler = optim.lr.PolynomialDecay(learning_rate=self.config.lr, decay_steps=ds, end_lr=1e-6)
        self.optimizer = optim.AdamW(self.scheduler, weight_decay=self.config.wd, parameters=self.net.parameters())

    def forward(self, inp):
        img = inp['img']
        mask = inp['mask']
        pred_mask = self.net(1 - img)
        loss = self.criterion(pred_mask, mask, img)
        return {
            'pred_mask': pred_mask,
            'loss': loss,
        }

    def train_epoch(self):
        self.net.train()
        losses = {"loss": AverageMeter()}
        for k, inp in tqdm.tqdm(enumerate(self.trainData), total=len(self.trainData), ncols=80):
            out = self.forward(inp)
            losses['loss'].update(out['loss'].item(), inp['img'].shape[0])
            out['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.clear_grad()
        return losses

    def eval_epoch(self, show=False):
        self.net.eval()
        losses = {"loss": AverageMeter()}
        for k, inp in tqdm.tqdm(enumerate(self.evalData), total=len(self.evalData), ncols=80):
            with paddle.no_grad():
                out = self.forward(inp)
                losses['loss'].update(out['loss'].item(), inp['img'].shape[0])
                reconImg = inp['img'] + (1 - inp['img']) * out['pred_mask']

                # show middle result through visdom
                if show and inp['img'].shape[0] == self.config.batchSize:
                    self.plotter.images(inp['img'] * 255, win='raw', opts=dict(title='raw'))
                    self.plotter.images(inp['gt'] * 255, win='gt', opts=dict(title='gt'))
                    self.plotter.images(inp['mask'] * 255, win='gt_mask', opts=dict(title='gt_mask'))
                    self.plotter.images(out['pred_mask'] * 255, win='pre_mask', opts=dict(title='pre_mask'))
                    self.plotter.images(reconImg * 255, win='pre', opts=dict(title='pre'))
        return losses

    def mask_pred(self, Trans, img):
        raw_size = img.shape[:-1]
        img = Trans(img)[None, ...]
        pred_mask = self.net(1 - img)
        pred_mask = F.interpolate(pred_mask, size=raw_size, mode='bicubic').cpu().numpy()[0]
        return pred_mask.transpose(1, 2, 0)

    def run(self):
        self.init_env()
        min_loss = math.inf
        for i in range(1, self.config.num_epochs + 1):
            train_result = self.train_epoch()

            info = f"Epoch {i}: "
            for k, v in train_result.items():
                info += f"{k}@{v():.5f}\t"
            info += f"lr:{self.optimizer.get_lr():.3e}"
            print(info)

            show = ((i % self.config.show_epoch) == 0) & (self.config.show_visdom)
            eval_result = self.eval_epoch(show)
            print(f"Eval: Loss {eval_result['loss']():.5f}")
            if min_loss > eval_result['loss']():
                if self.config.parallel:
                    best_model = copy.deepcopy(self.net.module.state_dict())
                else:
                    best_model = copy.deepcopy(self.net.state_dict())
        paddle.save(best_model,
                    os.path.join(self.config.modelsSavePath, self.config.arch, self.timestamp + '.pdparams'))

    def generate_mask(self, dataRoot, modelLog):
        import cv2
        import os

        from paddleseg.utils import utils
        from paddle.vision.transforms import Compose, ToTensor, Resize

        # load model
        net = MODEL[self.config.arch](num_classes=1)
        utils.load_entire_model(net, modelLog)
        self.net = net

        # load target file list
        with open(os.path.join(dataRoot, 'test.csv'), 'r') as f:
            imageFiles = [l.strip('\n') for l in f.readlines()]

        os.makedirs(os.path.join(dataRoot, 'pred_mask'), exist_ok=True)
        os.makedirs(os.path.join(dataRoot, 'result'), exist_ok=True)

        # start to generate mask
        self.net.eval()
        with paddle.no_grad():
            for imgName in tqdm.tqdm(imageFiles, ncols=80):
                img = cv2.imread(imgName)
                raw_img = img / 255
                raw_size = img.shape[:-1]

                img, square_mask = expand2square(img)

                crop_size = 1024
                if max(img.shape[:2]) > crop_size:
                    # get the expanding ratio. Non-linear function to compute the ratio.
                    ratio = min(np.log2(min(img.shape[:2]) / crop_size) + 1, 4)

                    # make the height/width be the multiple of 16
                    factor = 16
                    size = int(math.ceil(int(self.config.loadSize * ratio) / factor) * factor)
                    loadSize = (size,) * 2
                else:
                    loadSize = (self.config.loadSize,) * 2

                # resize and transfer the input to tensor
                Trans = Compose([
                    Resize(size=loadSize, interpolation='bicubic'),
                    ToTensor(),  # h * w * c --> c * h * w
                ])

                pred_mask = self.mask_pred(Trans, img)
                pred_mask = pred_mask[square_mask].reshape(*raw_size, 1)
                np.save(imgName.replace('images', 'pred_mask').replace('jpg', 'npy'), pred_mask)

                # compute the credit from the mean of result to judge the valid mask
                cred = min(np.mean(pred_mask[pred_mask > 0.01]), 0.25)
                pred_mask[pred_mask > cred] = 1
                reconImg = raw_img
                pred_mask = pred_mask[..., 0]

                # wipe the image through mask
                reconImg[pred_mask == 1] = 1
                cv2.imwrite(imgName.replace('images', 'result').replace('jpg', 'png'), reconImg * 255)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0,
                        help='1 shows to train the network')
    parser.add_argument('--arch', type=str, default='segformer_b2',
                        help='1 shows to train the network')
    parser.add_argument('--pretrained', type=int, default=1, help='1 shows pretrained the network')
    parser.add_argument('--modelLog', type=str,
                        default='Log/segformer_b1/01191623.pdparam', help='model saved place')

    parser.add_argument('--modelsSavePath', type=str, default='Log',
                        help='path for saving models')
    parser.add_argument('--logPath', type=str,
                        default='Log')
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--loadSize', type=int, default=384,
                        help='image loading size')
    parser.add_argument('--dataRoot', type=str,
                        default='data/dehw_train_dataset')
    parser.add_argument('--testDataRoot', type=str,
                        default='data/dehw_testA_dataset')

    parser.add_argument('--num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learnging rate')
    parser.add_argument('--ds', type=int, default=2000, help='learnging rate decay step')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')

    parser.add_argument('--numOfWorkers', type=int, default=16,
                        help='workers for dataloader')
    parser.add_argument('--parallel', type=int,
                        default=0, help='0 shows not')

    parser.add_argument('--show_visdom', type=int, default=0, help='0 represents not')
    parser.add_argument('--show_epoch', type=int, default=5, help='0 represents not')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    sess = Session(args)
    # sess.profile()
    if args.train:
        sess.run()
    else:
        sess.generate_mask(dataRoot=args.testDataRoot,
                           modelLog=args.modelLog)
