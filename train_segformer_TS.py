from dataset import newWetdataset, newWetdataset2, Wetlanddataset_TRG, WaterSupdataset
from dataset import MyLoader0520
import argparse
from tqdm import tqdm
from tabulate import tabulate
import torch
from torch import optim
import os
import torchvision
from torch.utils import data
from torch import nn
import numpy as np
# from model_fuse import FPN, fcn_resnet50, deeplabv3_resnet50
from backbones.Segformer import Segformer_baseline, Segformer_from_gsw
from dataset import build_semi_loader
import torch.nn.functional as F
from utils.loss_helper import compute_unsupervised_loss
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.pseudo_label import pseudolabel
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(100)  # 100


def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_dataset = newWetdataset2(txt='.\\data\\data-wetland0618\\train.txt',
                                   transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                                   loader=MyLoader0520)
    test_dataset = newWetdataset(txt='.\\data\\data-wetland0618\\test.txt',
                                 transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                                 loader=MyLoader0520)
    trg_dataset = WaterSupdataset(txt=args.trg_path,
                                  transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    print("Train numbers:{:d}".format(len(train_dataset)))
    print("Test numbers:{:d}".format(len(test_dataset)))
    print("Target numbers:{:d}".format(len(trg_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    trg_loader = torch.utils.data.DataLoader(trg_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = Segformer_baseline(5)
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    model_teacher = Segformer_baseline(5)
    model_teacher = model_teacher.to(device)

    # cost function
    cost = nn.CrossEntropyLoss().to(device)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    miou_max = 0.0
    trainer = Trainer(
        args,
        model,
        model_teacher,
        cost,
        optimizer,
        trg_loader,
        miou_max
    )

    for epoch in range(1, args.epochs + 1):

        trainer.training(epoch, train_loader)
        trainer.validation(epoch, test_loader)


class Trainer(object):
    def __init__(self, args, model, model_teacher, criterion, optimizer, trg_loader, miou_max):
        self.args = args
        self.model = model
        self.model_teacher = model_teacher
        self.cost = criterion
        self.optimizer = optimizer
        self.miou_max = miou_max
        self.trg_loader = trg_loader

    def training(self, epoch, train_loader):
        train_loss = 0.0
        self.model.train()

        # train_loader = tqdm(train_loader)
        trbar = tqdm(self.trg_loader)
        # trbar = self.target_loader
        num_img_tr = len(self.trg_loader)

        for i, data in enumerate(zip(train_loader, trbar)):
            i_iter = epoch * len(train_loader) + i
            source_set, target_set = data
            srimg, srlbl = source_set[0], source_set[1]
            trimg, trlbl = target_set[0], target_set[1]

            # 输入的原域和目标域的图像一块儿训练
            trimg = trimg.to(device)
            trimg = trimg.clone().detach().float()
            trlbl_water = trlbl.to(device, dtype=torch.int64)

            srimg = srimg.to(device)
            srimg = srimg.clone().detach().float()
            srlbl = srlbl.to(device, dtype=torch.int64)

            if epoch == 0:  # sup_only_epoch: 0
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                            self.model_teacher.parameters(), self.model.parameters()
                    ):
                        t_params.data = s_params.data

            # 生成伪标签
            self.model_teacher.eval()
            pred_u_teacher = self.model_teacher(trimg)
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u = torch.max(pred_u_teacher, dim=1)

            # forward
            allimg = torch.cat([srimg, trimg], dim=0)
            output = self.model(allimg)
            srout, trout = output.chunk(2, dim=0)  # chunk，分块

            # 有监督损失
            class_loss_sr = self.cost(srout, srlbl.squeeze(1))

            # teacher forward
            self.model_teacher.train()
            with torch.no_grad():
                out_t = self.model_teacher(allimg)
                pred_all_teacher = out_t
                srout_teacher, trout_teacher = pred_all_teacher.chunk(2, dim=0)

            # 伪标签损失计算
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - epoch / 200)
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                compute_unsupervised_loss(
                    trout,
                    label_u.clone(),
                    drop_percent,
                    trout_teacher.detach(),
                )
            )

            # 对目标域进行水体损失计算
            trout = torch.stack([trout[:, 0], trout[:, 1] + trout[:, 2] + trout[:, 3] + trout[:, 4]], dim=1)
            class_loss_tr = self.cost(trout, trlbl_water.squeeze(1))

            # 两种loss加和组成新的loss
            loss = class_loss_sr + class_loss_tr + unsup_loss
            # loss = class_loss_sr + unsup_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update teacher model with EMA
            with torch.no_grad():
                ema_decay = min(1 - 1 / (i_iter + 1), 0.99,)
                for t_params, s_params in zip(
                    self.model_teacher.parameters(), self.model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

            train_loss += loss.item()
            trbar.set_description('Epoch [%d/%d] Train loss: %.3f' % (epoch, args.epochs, train_loss / (i + 1)))

            # self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + trimg.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch, test_loader):
        self.model.eval()
        test_loss = 0.0
        # self.evaluator.reset()
        classes = ['背景', '河流', '湖泊', '季水', '养殖池']
        hist = torch.zeros(args.num_class, args.num_class).to(device)  # 混淆矩阵
        test_loader = tqdm(test_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):

                images = images.to(device)
                labels = labels.to(device, dtype=torch.int64).squeeze(1)
                images = images.clone().detach().float()
                outputs = self.model(images)
                loss = self.cost(outputs, labels)
                test_loss += loss
                test_loader.set_description(
                    'Epoch [%d/%d] Valid loss: %.3f' % (epoch, args.epochs, test_loss / (i + 1)))
                preds = outputs.softmax(dim=1).argmax(dim=1)
                keep = labels != 10  # 忽略的类别（低置信度类别）
                hist += torch.bincount(labels[keep] * args.num_class + preds[keep],
                                       minlength=args.num_class ** 2).view(args.num_class, args.num_class)

            ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
            miou = ious[~ious.isnan()].mean().item()
            ious = ious.cpu().numpy().tolist()
            Acc = hist.diag() / hist.sum(1)

            table = {
                'Class': classes,
                'IoU': ious,
                'Acc': Acc,
            }
            print('Validation:')
            print(tabulate(table, headers='keys'))
            print(f"\nOverall mIoU: {miou * 100:.2f}")

            if miou > self.miou_max:
                print('save new best miou', miou)
                torch.save(self.model, os.path.join(args.model_path, '{}.pth'.format(args.model_name)))
                self.miou_max = miou
                best_epoch = epoch

        # self.saver.save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': self.model.module.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'best_pred': self.best_pred,
        # }, is_best)
        # torch.save(self.model, os.path.join(args.model_path, 'cjzy-wetland-Segformer-{}-{}.pth'.format(args.cost, args.model_name)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--trg_path", default=r'', type=str)
    parser.add_argument("--data_path", default=r'', type=str)
    parser.add_argument("--num_class", default=5, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--cost", default='CE', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--model_name", default='l8-Segformer-TS+watersup', type=str)
    parser.add_argument("--model_path", default='./model/240618', type=str)
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        metavar='M', help='w-decay (default: 1e-5)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov')
    parser.add_argument('--factor', default=0.5, type=float,
                        help='ratio of pseudo-label')
    args = parser.parse_args()

    main(args)
