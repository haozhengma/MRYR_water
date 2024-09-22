import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from osgeo import gdal
from torchvision import transforms as T
import numpy as np
import cv2
import os
import random
import math
import glob
from utils.loss_helper import compute_unsupervised_loss

def MyLoader_GLH(path, type):

    if type == 'img':
        # return Image.open(path).convert('RGB')
        return np.array(Image.open(path))

    elif type == 'tif':
        # print(path)
        dataset = gdal.Open(path)
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数

        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).transpose(1, 2, 0).astype(np.float32)  # CHW->HWC
        return im_data

    elif type == 'label':
        label = np.array(Image.open(path)).astype(int)
        label[label == 255] = 1
        return label


# gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
def MyLoader0520(path, type):

    if type == 'img':
        # return Image.open(path).convert('RGB')
        return np.array(Image.open(path))

    elif type == 'tif':
        # print(path)
        dataset = gdal.Open(path)
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数

        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).transpose(1, 2, 0).astype(np.float32)  # CHW->HWC
        return im_data

    elif type == 'label':
        label = np.array(Image.open(path)).astype(int)
        return label


# 最初的dataloader
def MyLoader(path, type):
    if type == 'img':
        return Image.open(path).convert('RGB')
    elif type == 'tif':
        print(path)
        dataset = gdal.Open(path)
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数

        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).transpose(1, 2, 0)  # CHW->HWC
        # # 灰度图转换
        # im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  #HW->
        # im_data = np.expand_dims(im_data, 2)
        return im_data

    elif type == 'label':
        return np.array(Image.open(path)).astype(int)


class UVdataset(Dataset):
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        file = []
        with open(txt,'r', encoding='utf-8') as fh:
            for line in fh:
                # count = 0
                # print(line)
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split()
                # print(words)
                img_path = str(words[0])
                label_path = str(words[1])
                # print(imapath, labelpath)
                file.append((img_path, label_path))


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        img, label = self.file[index]
        # print(img, label)

        img = self.loader(img,type='tif')[:465, :465, :]
        label = self.loader(label,type='label')[:465, :465]

        # label[label==0] = 2
        # label = label-1
        # label[label==-1] = 0
        # label[label==2] = 0
        # label[label==3] = 2
        # label[label==4] = 3
        # label[label==5] = 4
        # print(img.shape, label.shape)

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)


class Wetlanddataset(Dataset):
    def __init__(self,txt, transform=None, target_transform=None, loader=MyLoader):
        file = []
        with open(txt,'r', encoding='utf-8') as fh:
            for line in fh:
                # count = 0
                # print(line)
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split()
                # print(words)
                img_path = str(words[0])
                label_path = str(words[1])
                # print(imapath, labelpath)
                file.append((img_path, label_path))

        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader

    def __getitem__(self,index):

        img, label = self.file[index]
        # print(img, label)

        img = self.loader(img,type='tif')[:465, :465, :]
        label = self.loader(label,type='label')[:465, :465]

        # label[label==0] = 2
        # label = label-1
        # label[label==-1] = 0
        # label[label==2] = 0
        # label[label==3] = 2
        # label[label==4] = 3
        # label[label==5] = 4
        # print(img.shape, label.shape)

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)


class Wetlanddataset_TRG(Dataset):
    def __init__(self, path, transform=None, target_transform=None, loader=MyLoader0520):
        files = os.listdir(path)
        file_list = [os.path.join(path, file) for file in files]
        self.file = file_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img = self.file[index]

        img = self.loader(img,type='tif')
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.file)


class WaterSupdataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader0520):
        file = []
        file_list = os.listdir(txt)
        for i in file_list:
            img_path = os.path.join(txt, i)
            label_path = os.path.join(txt, i).replace('images', 'labels')
            file.append((img_path, label_path))
        # with open(txt,'r', encoding='utf-8') as fh:
        #     for line in fh:
        #         # count = 0
        #         # print(line)
        #         line=line.strip('\n')
        #         line=line.rstrip()
        #         words=line.split()
        #         # print(words)
        #         img_path = str(words[0])
        #         label_path = str(words[1])
        #         # print(imapath, labelpath)
        #         file.append((img_path, label_path))

        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img, label = self.file[index]
        # print(img, label)

        img = self.loader(img,type='tif')
        label = self.loader(label,type='label')

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)


class glhdataset(Dataset):

    def __init__(self, path, transform=None, target_transform=None, loader=MyLoader_GLH):

        image_path = glob.glob(os.path.join(path, 'img') + "/*.jpg")
        label_path = glob.glob(os.path.join(path, 'label') + "/*.png")

        self.file_img = image_path
        self.file_lbl = label_path
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img = self.file_img[index]
        label = self.file_lbl[index]
        # print(img, label)

        img = self.loader(img, type='img')
        label = self.loader(label, type='label')

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file_img)


class newWetdataset(Dataset):

    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader0520):
        file = []
        with open(txt, 'r', encoding='utf-8') as fh:
            for line in fh:
                # count = 0
                # print(line)
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                # print(words)
                img_path = str(words[0])
                label_path = str(words[1])
                # print(imapath, labelpath)
                file.append((img_path, label_path))

        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img, label = self.file[index]
        # print(img, label)

        img = self.loader(img, type='tif')
        label = self.loader(label, type='label')

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)


class newWetdataset2(Dataset):

    def __init__(self, txt, n_trg=882, transform=None, target_transform=None, loader=MyLoader0520):
        file = []
        with open(txt, 'r', encoding='utf-8') as fh:
            for line in fh:
                # count = 0
                # print(line)
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                # print(words)
                img_path = str(words[0])
                label_path = str(words[1])
                # print(imapath, labelpath)
                file.append((img_path, label_path))

        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if len(self.file) < n_trg:
            num_repeat = math.ceil(n_trg / len(self.file))
            self.list_sample = self.file * num_repeat
            self.list_sample_new = random.sample(self.list_sample, n_trg)
        else:
            self.list_sample_new = self.file

    def __getitem__(self, index):

        img, label = self.list_sample_new[index]
        # print(img, label)

        img = self.loader(img, type='tif')
        label = self.loader(label, type='label')

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.list_sample_new)


class supDataset(Dataset):

    def __init__(self, txt, transform=None, split='train', loader=MyLoader0520):
        file = []
        with open(txt, 'r', encoding='utf-8') as fh:
            for line in fh:
                img_path = line.strip()
                lbl_path = img_path.replace('images', 'labels')
                file.append((img_path, lbl_path))

        self.file = file
        self.transform = transform
        self.loader = loader

        if len(self.file) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.file))
            self.list_sample = self.file * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

    def __getitem__(self, index):
        img, label = self.file[index]
        image = self.loader(img, type='tif')
        label = self.loader(label, type='label')
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


class unsupDataset(Dataset):

    def __init__(self, txt, transform=None, loader=MyLoader0520):
        file = []
        with open(txt, 'r', encoding='utf-8') as fh:
            for line in fh:
                img_path = line.strip()
                lbl_path = img_path.replace('images', 'labels')
                file.append((img_path, lbl_path))

        self.file = file
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img = self.file[index]
        image = self.loader(img, type='tif')

        image,  = self.transform(image)
        return image[0]

    def __len__(self):
        return len(self.list_sample_new)


def build_semi_loader(args, split, seed=0):

    batch_size = args.batch_size
    n_unsup = 2975 - 744  # n_sup: 744

    # build transform
    trs_form = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if split == "val":

        dset = supDataset(args.data_path,
                          args.labeled_path,
                          trs_form,
                          seed,
                          n_unsup,
                          split)

        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:

        dset = supDataset(args,
                          trs_form,
                          seed,
                          n_unsup,
                          split)

        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        dset_unsup = unsupDataset(args,
                                  trs_form,
                                  seed,
                                  n_unsup)

        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup


if __name__ == "__main__":

    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test = glhdataset(r'I:\data\glhwater\train', transform=test_transform)
    print(test)
    train_loader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=True, pin_memory=True)
    print(train_loader)
    # test_dataset=UVdataset(txt='.\\data\\testUV.txt',transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True)
    # for step,(img, label) in enumerate(test_loader):
    #     print(img.shape, label[:, :, 300, 300])
    # test = WaterSupdataset(r'I:\data\landsat_labelme\test2\trgimages', transform=test_transform)
    # test_dataset = newWetdataset(txt='data/data-wetland0520/test.txt', transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    # for step,(img, label) in enumerate(test_loader):
    #     print(img.shape, label[:, :, 300, 300])

    # trg_dataset = Wetlanddataset_TRG(transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),loader=MyLoader10)

