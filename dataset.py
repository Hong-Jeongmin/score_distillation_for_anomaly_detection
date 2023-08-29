import torch, torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
import numpy as np
import os

BICUBIC = InterpolationMode.BICUBIC
from copy import deepcopy

Data_Path = "./data"
if not os.path.isdir(Data_Path):
    os.mkdir(Data_Path)

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])



transform_csi = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

def get_transform(args):

    if (args.teacher == "CSI") & (args.mode =='train') :
        if args.dataset == "fashion":
            return transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                ])
        
        return transforms.ToTensor()
    
    elif args.dataset == "fashion":
        return transform_gray
 
    else:
        return transform_color

def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]




def get_outliers_loader(args):
    transform = get_transform(args)

    if args.dataset == "fashion":
        dataset = torchvision.datasets.EMNIST(root = Data_Path, split = 'letters', train=True, transform = transform, download = True)
    else:
        dataset = torchvision.datasets.ImageFolder(root=Data_Path+'/tiny', transform=transform)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return outlier_loader

from PIL import Image
class OE_Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.image = np.load(path)
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        x = self.image[idx]

        if self.transform:
            x = Image.fromarray(x)
            x = self.transform(x)     
        return x
        

def get_loaders(args):
    
    transform = get_transform(args)
    if args.dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        coarse = {}
        trainset = ds(root=Data_Path, train=True, download=True, transform=transform, **coarse)
        testset = ds(root=Data_Path, train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        ds = torchvision.datasets.CIFAR100
        testset = ds(root=Data_Path,
                           train=False, download=True,
                           transform=transform)

        trainset = ds(root=Data_Path,
                            train=True, download=True,
                            transform=transform)

        trainset.targets = sparse2coarse(trainset.targets)
        testset.targets = sparse2coarse(testset.targets) 
    
    elif args.dataset == "fashion":
        ds = torchvision.datasets.FashionMNIST
 
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
   
    label_class = args.label
    idx = [i for i, t in enumerate(trainset.targets) if (t == label_class)]
    testset.targets = [int(t != label_class) for t in testset.targets]
    # trainset.data = trainset.data[idx]
    # trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
    trainset = Subset(trainset, idx)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, test_loader


