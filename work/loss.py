import paddle
import numpy as np
import paddle.nn as nn

def muti_loss_fusion(pred, labels_v, region, func=nn.BCELoss(reduction='mean')):
    losses = []
    for p in pred:
        losses.append(func(p[region], labels_v[region]))
    loss = sum(losses)
    return loss

def gram_matrix(feat):
    # https://github.com/pypaddle/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.shape
    feat = feat.reshape((b, ch, h * w))
    feat_t = feat.transpose(1, 2)
    gram = paddle.bmm(feat, feat_t) / (ch * h * w)
    return gram

def dice_loss(input, target):
    # input = paddle.sigmoid(input)

    input = input.reshape((input.shape[0], -1))
    target = target.reshape((target.shape[0], -1))
    
    input = input 
    target = target

    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1 - dice_loss

class PSNRLoss(nn.Layer):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = paddle.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.shape) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.
            target = (target * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.shape) == 4

        return self.loss_weight * self.scale * paddle.log(((pred - target) ** 2).mean(axis=(1, 2, 3)) * 255 + 1).mean()
        # return self.loss_weight * ((pred - target) ** 2).mean(dim=(1, 2, 3)).mean()


class MaskLoss(nn.Layer):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.entropy_loss = nn.BCELoss()
    
    def forward(self, pred_mask, mask, img):
        loss = 0
        # loss += dice_loss(pred_mask, mask)
        # loss += self.entropy_loss(pred_mask, mask)
        # # loss += 10 * self.l1_loss(pred_mask, mask)

        # focus on the words part where the pixle values are usually small
        valid_region = paddle.min(img, axis=1, keepdim=True) < 0.5
        loss += dice_loss(pred_mask[valid_region], mask[valid_region])
        loss += self.entropy_loss(pred_mask[valid_region], mask[valid_region])

        loss += .8 * dice_loss(pred_mask[~valid_region], mask[~valid_region])
        loss += .8 * self.entropy_loss(pred_mask[~valid_region], mask[~valid_region])
        return loss