import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data
from glob import glob
import cv2
import os
import numpy as np
import datetime, time
import re


class TrainDataset(data.Dataset):
    def __init__(self, root=''):
        super(TrainDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image_pre', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask_pre', basename[:-4] + '_mask.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        data = np.array([data, data, data])
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


class TestDataset(data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        data = np.array([data, data, data])
        return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.img_files)


def categorical_dice(mask1, mask2, label_class=1):
    """
    Dice score of a specified class between two volumes of label masks.
    (classes are encoded but by label class number not one-hot )
    Note: stacks of 2D slices are considered volumes.

    Args:
        mask1: N label masks, numpy array shaped (H, W, N)
        mask2: N label masks, numpy array shaped (H, W, N)
        label_class: the class over which to calculate dice scores

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).float()
    mask2_pos = (mask2 == label_class).float()
    dice = 2 * (mask1_pos * mask2_pos).sum() / (mask1_pos.sum() + mask2_pos.sum())
    return dice


class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        losses = {}
        targets_view = targets.view(-1)
        for name, x in inputs.items():

            x_view = x.argmax(1)
            x_view = x_view.view(-1)

            dice_loss = 0.0
            for i in range(4):
                # tmp_1 = torch.zeros(batch_size*96*96)
                # tmp_2 = torch.zeros(batch_size*96*96)
                # for idx, (j, k) in enumerate(zip(x_view, targets_view)):
                #     tmp_1[idx] = 1 if j.int() == i else 0
                #     tmp_2[idx] = 1 if k.int() == i else 0
                tmp_1 = (x_view == i).float()
                tmp_2 = (targets_view == i).float()
                intersection = (tmp_1 * tmp_2).sum()
                dice_loss += 1 - (2. * intersection + smooth) / (tmp_1.sum() + tmp_2.sum() + smooth)
            dice_loss_avg = dice_loss / 4.
            CE = F.cross_entropy(x, targets.long())
            Dice_CE = CE + dice_loss_avg

            losses[name] = Dice_CE

        if len(losses) == 1:
            return losses['out']
        return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    """
    Predict the masks of validation datasets by using current model and calculate the overall dice and dices for separate classes.
    :param model: current model and weights
    :param data_loader: the data_loader should contain validation data
    :param device: device to run current programme
    :param num_classes: number of classes needed to be segmented
    :return: overall dice and dices for each class
    """

    dice_avg = 0.
    dice_class = [0., 0., 0., 0.]
    cnt = 0

    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            output = output.argmax(1).squeeze(0)
            dice = 0.
            for i in range(num_classes):
                dice_class[i] += categorical_dice(output, target, i)
                if i != 0:
                    dice += categorical_dice(output, target, i)
            dice /= 3.
            # print(dice)
            dice_avg += dice
            cnt += 1
    dice_avg /= cnt
    for i in range(num_classes):
        dice_class[i] = dice_class[i].item() / cnt
    return dice_avg, dice_class


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int):
    assert num_step > 0 and epochs > 0

    def f(x):
        return (1 - (x - num_step) / ((epochs - 1) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    num_workers = 0
    batch_size = 5
    num_classes = 4
    epoch_num = 200
    aux = True
    pretrain = True

    train_data_path = './data/train'
    train_set = TrainDataset(train_data_path)
    train_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    val_data_path = './data/val'
    val_set = TrainDataset(val_data_path)
    val_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=1, shuffle=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False, num_classes=num_classes,
                           aux_loss=True)
    model.to(device)

    if pretrain:
        weights_dict = torch.load("./deeplabv3_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        model.load_state_dict(weights_dict, strict=False)

    loss_func = DiceCELoss()
    loss_func.to(device)

    init_lr = 0.001

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    if aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": init_lr * 10})

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=init_lr, weight_decay=1e-4
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.995)
    txt_name = 'result' + str(int(time.time())) + '.txt'
    f = open(txt_name, 'w+')

    for epoch in range(epoch_num):
        loss_avg = 0.
        model.train()
        print('epoch[{}]  ###########'.format(epoch))
        for image, target in train_data_loader:
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = loss_func(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            loss_avg += loss.item()

        loss_avg /= 1. * len(train_data_loader.dataset) / batch_size

        val_dice, val_dice_class = evaluate(model, val_data_loader, device, num_classes)
        print_txt = 'epoch[{}], train loss={}, val dice={}, val dice class={}, lr={}'.format(epoch, loss_avg, val_dice,
                                                                                             val_dice_class, lr)
        print(print_txt)
        f.write(print_txt + '\n')

        save_file = {
            "model": model.state_dict(),
            "epoch": epoch,
        }

        torch_path = r"C:\Users\quchenyuan\Desktop\weights/model_" + str(epoch) + ".pth"
        torch.save(save_file, torch_path)

    train_loss = []
    for line in f:
        current_train_loss = re.match('train loss=\d*,', line)
        train_loss.append(current_train_loss)

    f.close()


if __name__ == '__main__':
    train()
