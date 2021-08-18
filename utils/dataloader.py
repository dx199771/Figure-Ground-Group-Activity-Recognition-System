"""
    Author: Xu Dong
    Student Number: 200708160
    Email: x.dong@se20.qmul.ac.uk

    School of Electronic Engineering and Computer Science
    Queen Mary University of London, UK
    London, UK
"""
import torch
from torch.utils.data import Dataset

from PIL import Image
import os, pickle ,math
import numpy as np
import utils.util as utils
import Config

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

def read_txt(path):
    text_file = open(path, "r")
    lines = text_file.readlines()
    return lines

def get_label(label_file,label):
    for i in label_file:
        if(i.split()[0] == label[:-4]):
            return i.split()[1:-1]
    return None

class Twostream_dataloader(Dataset):
    def __init__(self, img_folder_path, skeleton_folder_path,txt_file_path, num_classes, transforms=None):
        self.img_folder_path = img_folder_path
        self.skeleton_folder_path = skeleton_folder_path

        self.index_lookup_img = os.listdir(self.img_folder_path)
        self.index_lookup_img.sort(key=lambda x: int(x))

        self.index_lookup_skeleton = os.listdir(self.skeleton_folder_path)

        self.skeleton_folder_path = skeleton_folder_path
        self.index_lookup = os.listdir(self.skeleton_folder_path)
        self.index_lookup.sort(key=lambda x: int(x[:-4]))
        self.data_label = read_txt(txt_file_path)
        self.num_classes = num_classes
        self.transforms = transforms
    def __getitem__(self, index):
        data = []

        img_path = self.index_lookup_img[index]

        img_list = os.listdir(os.path.join(self.img_folder_path, img_path))
        img_list.sort()
        img_list.sort(key=lambda x: int(x.split("_")[1][:-4]))
        for image in img_list:
            img = Image.open(os.path.join(self.img_folder_path, img_path, image))
            label = image.split("_")[0]
            if self.transforms is not None:
                img = self.transforms(img)
            data.append(img)
        inputs = torch.stack(data).permute(1, 0, 2, 3)

        npy_data = os.path.join(self.skeleton_folder_path, self.index_lookup[index])
        data = np.load(npy_data, allow_pickle=True)
        label_skeleton = get_label(self.data_label, self.index_lookup[index])
        if data.shape[0] == 0:
            data_ = np.zeros((8, 25, 3))

            return inputs[:,::2,:,:], torch.from_numpy(utils.index_to_onehot(label, self.num_classes)),\
                   torch.from_numpy(data_[:, :, :2]), utils.index_to_onehot(label_skeleton, self.num_classes, figure=True)
        data_ = skeleton_interpolation(data)

        return inputs[:,::2,:,:], torch.from_numpy(utils.index_to_onehot(label, self.num_classes)),\
               torch.from_numpy(data_[:, :, :2]), utils.index_to_onehot(label_skeleton, self.num_classes, figure=True)

    def __len__(self):
        return len(self.index_lookup)


class Groundstream_dataloader(Dataset):
    def __init__(self,folder_path, txt_file_path , num_classes, transforms=None):
        self.img_folder_path = folder_path
        self.index_lookup = os.listdir(self.img_folder_path)
        self.data_label = read_txt(txt_file_path)
        self.transforms = transforms
        self.num_classes = num_classes

    def __getitem__(self, index):
        data = []
        path = self.index_lookup[index]

        # sort image name by seq
        img_list = os.listdir(os.path.join(self.img_folder_path,path))
        img_list.sort()
        img_list.sort(key = lambda  x: int(x.split("_")[1][:-4]))
        for image in img_list:
            img = Image.open(os.path.join(self.img_folder_path,path,image))
            label = image.split("_")[0]
            if self.transforms is not None:
                img = self.transforms(img)
            data.append(img)
        inputs = torch.stack(data).permute(1,0,2,3)

        return (inputs[:,::2,:,:], torch.from_numpy(utils.index_to_onehot(label, self.num_classes)))

    def __len__(self):
        return len(self.index_lookup)

class Figurestream_dataloader(Dataset):
    def __init__(self,folder_path, txt_file_path, num_classes, transforms=None):
        self.img_folder_path = folder_path
        self.index_lookup = os.listdir(self.img_folder_path)
        self.index_lookup.sort()
        self.index_lookup.sort(key=lambda x: int(x[:-4]))
        self.data_label = read_txt(txt_file_path)

        #self.index_lookup = os.listdir(self.img_folder_path).sort(key=lambda x: int(x[:-4]))
        #self.data_label = self.read_txt(txt_file_path)
        self.num_classes = num_classes

        self.transforms = transforms

    def __getitem__(self, index):
        npy_data = os.path.join(self.img_folder_path, self.index_lookup[index])
        data = np.load(npy_data, allow_pickle=True)
        label = get_label(self.data_label,self.index_lookup[index])

        if data.shape[0] == 0:
            data_ = np.zeros((8,25,3))

            return torch.from_numpy(data_[:,:,:2]) , utils.index_to_onehot(label, self.num_classes,figure=True)
        data_ = skeleton_interpolation(data)
        #print(utils.index_to_onehot(label, self.num_classes))
        return torch.from_numpy(data_[:,:,:2]) , utils.index_to_onehot(label, self.num_classes,figure=True)
    def __len__(self):
        return len(self.index_lookup)

def skeleton_interpolation(data):
    # eliminate none objects
    if data.shape[0] == 8:
        return data
    elif data.shape[0] > 8:
        # if number of skeleton data larger than 8
        # find 8 maximum values
        argmax = np.argpartition((np.sum(data,axis=1)[:,-1]), -8)[-8:]
        return data[argmax]
    elif data.shape[0]<8:
        # if number of skeleton data less than 8
        # do data generation
        return np.repeat(data,[math.ceil(8/data.shape[0])],axis=0)[:8]




"""
for i in os.listdir(r"F:\Disssertation\single_test\data\SkeletonData"):
    a = np.load(os.path.join(r"F:\Disssertation\single_test\data\SkeletonData", i),allow_pickle=True)
    if a.ndim == 1:
        if len(a) != 0:
            print(a[0].shape, a[0])
        #print(np.concatenate(a,axis=0).shape,a.shape)

    #print(a.shape)
"""