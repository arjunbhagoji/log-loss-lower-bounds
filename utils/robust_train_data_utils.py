import os
import scipy
import scipy.io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle

# ImageNet type datasets, i.e., which support loading with ImageFolder
def imagenette(datadir="./data/imagenette", batch_size=128, mode="org", size=224, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       norm_layer
                       ])
    transform_test = transforms.Compose([transforms.Resize(int(1.14*size)),
                      transforms.CenterCrop(size),
                      transforms.ToTensor(), 
                      norm_layer])
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.ImageFolder(
        os.path.join(datadir, "train"), 
        transform=transform_train)
    testset = datasets.ImageFolder(
        os.path.join(datadir, "val"), 
        transform=transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def cifar10(datadir="./data/", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def update_list(vals, indices, c):
    for i in indices:
        vals[i] = c
    return vals


def cifar_3_7(datadir="./data/", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, args=None, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    # extract images of class 3 and 7
    indices_train_3, indices_train_7 = np.where(np.array(trainset.targets)==3)[0], np.where(np.array(trainset.targets)==7)[0]
    indices_train = np.concatenate([indices_train_3, indices_train_7])
    if args.opt_probs:
        f = f"./optimal_probs/logloss_3_7_5000_CIFAR-10_l2_{args.epsilon}.txt"
        soft_labels = [[float(c) for c in line.split()] for line in open(f, "r").read().split("\n") if line != ""]
        print(f"Loading soft labels from {f}")
        for i, index in enumerate(indices_train):
            trainset.targets[index] = torch.tensor(soft_labels[i])
    else:
        trainset.targets = update_list(trainset.targets, indices_train_3, 0)
        trainset.targets = update_list(trainset.targets, indices_train_7, 1)
    
    indices_test_3, indices_test_7 = np.where(np.array(testset.targets)==3)[0], np.where(np.array(testset.targets)==7)[0]
    indices_test = np.concatenate([indices_test_3, indices_test_7])
    testset.targets = update_list(testset.targets, indices_test_3, 0)
    testset.targets = update_list(testset.targets, indices_test_7, 1)
    
    print("No support for distributed sampler (thus --ddp multi-gpu mode is not supported)")
    train_sampler = torch.utils.data.SubsetRandomSampler(indices_train)
    test_sampler = torch.utils.data.SubsetRandomSampler(indices_test)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def clip_soft_labels(soft_labels, clip=0.6):
    """
        Clip soft-labels to avoid class flipping. 
        For eg., if soft-labels for an class-0 images are [0.45, 0.55] we clip it to [0.6, 0.4]
    """
    print("Assuming first-half images belong to class-0 and next-half to class-1")
    n = len(soft_labels)
    h1 = [s if s[0] > 0.5 else [clip, 1-clip] for s in soft_labels[:n//2]]
    h2 = [s if s[1] > 0.5 else [1-clip, clip] for s in soft_labels[n//2:]]
    return h1+h2


def drop_soft_labels(soft_labels):
    """
        set soft-labels to [0, 0] to avoid class flipping. 
        For eg., if soft-labels for an class-0 images are [0.45, 0.55] we set it to [0., 0.]
    """
    print("Assuming first-half images belong to class-0 and next-half to class-1")
    n = len(soft_labels)
    h1 = [s if s[0] > 0.5 else [0., 0.] for s in soft_labels[:n//2]]
    h2 = [s if s[1] > 0.5 else [0., 0.] for s in soft_labels[n//2:]]
    return h1+h2


class CustomLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, labels):
        isinstance(indices, (np.ndarray, np.generic) )
        self.base_dataset = base_dataset
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if idx in self.indices:
            label = torch.tensor(self.labels[np.where(self.indices==idx)[0][0]])
        return img, label
    
    
def mnist_3_7(datadir="./data/", batch_size=128, mode="org", size=28, normalize=False, norm_layer=None, workers=4, distributed=False, args=None, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.], std=[0.]) # wrong mean/std
        else:
            norm_layer = transforms.Normalize(mean=[0.], std=[1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.ToTensor(), norm_layer]
    if size != 28:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 28:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.MNIST(
            root=os.path.join(datadir, "mnist"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.MNIST(
            root=os.path.join(datadir, "mnist"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    # extract images of class 3 and 7 
    # selecting only first 5k samples as only these have soft-labels
    trainset.targets = list(trainset.targets.numpy()) # cause each label can either be int or one-hot now
    indices_train_3, indices_train_7 = np.where(np.array(trainset.targets)==3)[0][:5000], np.where(np.array(trainset.targets)==7)[0][:5000]
    indices_train = np.concatenate([indices_train_3, indices_train_7])
    if args.opt_probs:
        f = f"./optimal_probs/logloss_3_7_5000_MNIST_l2_{args.epsilon}.txt"
        soft_labels = [[float(c) for c in line.split()] for line in open(f, "r").read().split("\n") if line != ""]
        print(f"Loading soft labels from {f}")
        if args.clip_soft_labels:
            soft_labels = clip_soft_labels(soft_labels, clip=0.6)
        if args.drop_soft_labels:
            soft_labels = drop_soft_labels(soft_labels)
        trainset =  CustomLabelDataset(trainset, indices_train, soft_labels) # update the dataset itself (hack of updating label in original dataset no working)
    else:
        trainset.targets = update_list(trainset.targets, indices_train_3, 0)
        trainset.targets = update_list(trainset.targets, indices_train_7, 1)
    
    indices_test_3, indices_test_7 = np.where(np.array(testset.targets)==3)[0], np.where(np.array(testset.targets)==7)[0]
    indices_test = np.concatenate([indices_test_3, indices_test_7])
    testset.targets = update_list(testset.targets, indices_test_3, 0)
    testset.targets = update_list(testset.targets, indices_test_7, 1)
    
    print("No support for distributed sampler (thus --ddp multi-gpu mode is not supported)")
    train_sampler = torch.utils.data.SubsetRandomSampler(indices_train)
    test_sampler = torch.utils.data.SubsetRandomSampler(indices_test)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def fmnist_3_7(datadir="./data/", batch_size=128, mode="org", size=28, normalize=False, norm_layer=None, workers=4, distributed=False, args=None, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.], std=[0.]) # wrong mean/std
        else:
            norm_layer = transforms.Normalize(mean=[0.], std=[1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.ToTensor(), norm_layer]
    if size != 28:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 28:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.FashionMNIST(
            root=os.path.join(datadir, "fmnist"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.FashionMNIST(
            root=os.path.join(datadir, "fmnist"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    # extract images of class 3 and 7 
    # selecting only first 5k samples as only these have soft-labels
    trainset.targets = list(trainset.targets.numpy()) # cause each label can either be int or one-hot now
    indices_train_3, indices_train_7 = np.where(np.array(trainset.targets)==3)[0][:5000], np.where(np.array(trainset.targets)==7)[0][:5000]
    indices_train = np.concatenate([indices_train_3, indices_train_7])
    if args.opt_probs:
        f = f"./optimal_probs/logloss_3_7_5000_fMNIST_l2_{args.epsilon}.txt"
        soft_labels = [[float(c) for c in line.split()] for line in open(f, "r").read().split("\n") if line != ""]
        print(f"Loading soft labels from {f}")
        if args.clip_soft_labels:
            soft_labels = clip_soft_labels(soft_labels, clip=0.6)
        if args.drop_soft_labels:
            soft_labels = drop_soft_labels(soft_labels)
        trainset =  CustomLabelDataset(trainset, indices_train, soft_labels) # update the dataset itself (hack of updating label in original dataset no working)
    else:
        trainset.targets = update_list(trainset.targets, indices_train_3, 0)
        trainset.targets = update_list(trainset.targets, indices_train_7, 1)
    
    indices_test_3, indices_test_7 = np.where(np.array(testset.targets)==3)[0], np.where(np.array(testset.targets)==7)[0]
    indices_test = np.concatenate([indices_test_3, indices_test_7])
    testset.targets = update_list(testset.targets, indices_test_3, 0)
    testset.targets = update_list(testset.targets, indices_test_7, 1)
    
    print("No support for distributed sampler (thus --ddp multi-gpu mode is not supported)")
    train_sampler = torch.utils.data.SubsetRandomSampler(indices_train)
    test_sampler = torch.utils.data.SubsetRandomSampler(indices_test)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train

