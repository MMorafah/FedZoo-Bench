import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from .datasetzoo import DatasetZoo

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
def get_transforms(dataset, noise_level=0, net_id=None, total=0):
    if dataset == 'mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    elif dataset == 'usps':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
         ])

    elif dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
         ])

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

    elif dataset == 'stl10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif dataset == 'femnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
            transforms.Normalize((0.1307,), (0.3081,))
         ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    return transform_train, transform_test

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(np.sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class DatasetKD(Dataset):
    def __init__(self, dataset, logits):
        self.dataset = dataset
        self.logits = logits
    
    def set_logits(self, logits):
        self.logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = self.logits[item]
        return image, label, logits
    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self._make_data()
        
    def _make_data(self):
        self.data, self.target = [], []
        for idx in self.idxs:
            tmp_data, tmp_target = self.dataset[idx]
            tmp_data = tmp_data.numpy()
            if tmp_data.shape[1] == tmp_data.shape[2]:
                tmp_data = np.transpose(tmp_data, [1,2,0])
            self.data.append(tmp_data)
            self.target.append(tmp_target)
        #print(tmp_data.shape)
        #print(tmp_target.shape)
        self.data = torch.Tensor(np.array(self.data))
        self.target = torch.Tensor(np.array(self.target))
        #print(self.data.shape)
        #print(self.target.shape)
        return
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
     
def get_subset(dataset, idxs): 
    return DatasetSplit(dataset, idxs)

def get_dataset_global(dataset, datadir, batch_size=128, p_train=1.0, p_test=1.0):
    transform_train, transform_test = get_transforms(dataset, noise_level=0, net_id=None, total=0)
    
    train_ds_global = DatasetZoo(datadir, dataset=dataset, dataidxs=None, train=True, 
                                 transform=transform_train, target_transform=None, download=True, p_data=p_train)
    
    test_ds_global = DatasetZoo(datadir, dataset=dataset, dataidxs=None, train=False, 
                                 transform=transform_train, target_transform=None, download=True, p_data=p_test)
    
    train_dl_global = DataLoader(dataset=train_ds_global, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl_global = DataLoader(dataset=test_ds_global, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_ds_global, test_ds_global, train_dl_global, test_dl_global