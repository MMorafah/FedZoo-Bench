import numpy as np
from torchvision import datasets 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from .custom_data import *

class DatasetZoo(Dataset):
    def __init__(self, root, dataset='cifar10', dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False, p_data=1.0):
        
        self.root = root
        self.dataset = dataset
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.p_data = p_data
        
        self.data, self.target, self.dataobj, self.mode = self.__init_dataset__()

    def __init_dataset__(self):
        
        if self.dataset == 'mnist':
            dataobj = datasets.MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'
        elif self.dataset == 'usps':
            dataobj = datasets.USPS(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'
        elif self.dataset == 'fmnist':
            dataobj = datasets.FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'
        elif self.dataset == 'cifar10':
            dataobj = datasets.CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'cifar100':
            dataobj = datasets.CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'svhn':
            if self.train:
                dataobj = datasets.SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = datasets.SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'stl10':
            if self.train:
                dataobj = datasets.STL10(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = datasets.STL10(self.root, 'test', self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'celeba':
            X_train, y_train, X_test, y_test = load_celeba_data(datadir)
            mode = 'RGB'
        elif self.dataset == 'tinyimagenet':
            dataobj = load_tinyimagenet_data(self.root, self.train)
            mode = 'RGB'
        elif self.dataset == 'femnist':
            dataobj = datasets.FEMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'

        data = np.array(dataobj.data)
        try:
            target = np.array(dataobj.targets)
        except:
            target = np.array(dataobj.labels)

        if len(data.shape) > 2 and data.shape[2]==data.shape[3]:
            data = data.transpose(0,2,3,1) ## STL-10

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            
        if self.dataidxs is None: 
            idxs_data = np.arange(len(data))
            idxs_target = np.arange(len(target))

            perm_data = idxs_data  #np.random.permutation(idxs_data) 

            p_data1 = int(len(idxs_data)*self.p_data)
            perm_data = perm_data[0:p_data1]

            data = data[perm_data] 
            target = target[perm_data]

        return data, target, dataobj, mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if len(img.shape) == 1: ## tinyimagenet
            img = self.dataobj.loader(img[0])
        elif img.shape[1]==img.shape[2]:  ## SVHN
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        else:
            img = Image.fromarray(img, mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target.type(torch.LongTensor)

        return img, target

    def __len__(self):
        return len(self.data)

def load_tinyimagenet_data(datadir, train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    if train:
        ds = ImageFolder(datadir+'tiny-imagenet-200/train/', transform=transform)
    else:
        ds = ImageFolder(datadir+'tiny-imagenet-200/val/', transform=transform)
    ds.data = ds.imgs
    ds.target = ds.targets
    return ds