import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .mnist_custom_utils import MNIST
from .cifar_custom_utils import cifar10


# def data_augmentation(input_images, max_rot=25, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2):
#     def sometimes(aug): return iaa.Sometimes(0.5, aug)
#     seq = iaa.Sequential([
#         iaa.Fliplr(0.3),  # horizontally flip 50% of the images
#         # iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
#         sometimes(iaa.Affine(
#             # scale images to 80-120% of their size, individually per axis
#             scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#             # translate by -20 to +20 percent (per axis
#             translate_percent={"x": (-width_shift_range, width_shift_range),
#                                "y": (-height_shift_range, height_shift_range)},
#             cval=(0, 1),  # if mode is constant, use a cval between 0 and 255
#         )),
#         sometimes(iaa.Affine(
#             rotate=(-max_rot, max_rot),  # rotate by -45 to +45 degrees
#         ))
#     ])
#     return seq.augment_images(input_images)


def load_dataset(args, data_dir):
    if args.dataset_in == 'CIFAR-10':
        loader_train, loader_test, data_details = load_cifar_dataset(args, data_dir)
    elif args.dataset_in == 'MNIST':
        loader_train, loader_test, data_details = load_mnist_dataset(args, data_dir)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test, data_details


def load_mnist_dataset(args, data_dir):
    # MNIST data loaders
    trainset = datasets.MNIST(root=data_dir, train=True,
                                download=False, 
                                transform=transforms.ToTensor())
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.MNIST(root=data_dir,
                                train=False,
                                download=False, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    return loader_train, loader_test, data_details


def load_cifar_dataset(args, data_dir):
    # CIFAR-10 data loaders
    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                download=False, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()
                                ]))
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.CIFAR10(root=data_dir,
                                train=False,
                                download=False, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}
    return loader_train, loader_test, data_details


def load_dataset_custom(args, data_dir):
    if args.dataset_in == 'CIFAR-10':
        loader_train, loader_test, data_details = load_cifar_dataset_custom(args, data_dir)
    elif args.dataset_in == 'MNIST':
        loader_train, loader_test, data_details = load_mnist_dataset_custom(args, data_dir)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test, data_details

def load_mnist_dataset_custom(args, data_dir, training=True):
    # MNIST data loaders
    trainset = MNIST(root=data_dir, args=args, train=True,
                                download=False, 
                                transform=transforms.ToTensor(),
                                dropping=args.dropping,
                                training=training)
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = MNIST(root=data_dir, args=args, train=False,
                                download=False, 
                                transform=transforms.ToTensor(),
                                dropping=args.dropping,
                                training=False)
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    return loader_train, loader_test, data_details

def load_cifar_dataset_custom(args, data_dir):
    # CIFAR-10 data loaders
    trainset = cifar10(root=data_dir, args=args, train=True,
                                download=False, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()
                                ]),
                                dropping=args.dropping)
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = cifar10(root=data_dir, args=args,
                                train=False,
                                download=False, 
                                transform=transforms.ToTensor(),
                                dropping=args.dropping)

    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}
    return loader_train, loader_test, data_details

def load_dataset_numpy(args, data_dir):
    if args.dataset_in == 'MNIST':
        trainset = MNIST(root=data_dir, args=args, train=True,
                            download=False,
                            np_array=True)
        testset = MNIST(root=data_dir, args=args, train=False,
                                download=False,
                                np_array=True)
    elif args.dataset_in == 'CIFAR-10':
        trainset = cifar10(root=data_dir, args=args, train=True,
                            download=False, 
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                            ]))
        testset = cifar10(root=data_dir, args=args,
                                train=False,
                                download=False)
    return trainset, testset