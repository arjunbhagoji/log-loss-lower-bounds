import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


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
        loader_train, loader_test = load_cifar_dataset(args, data_dir)
    elif args.dataset_in == 'MNIST':
        loader_train, loader_test = load_mnist_dataset(args, data_dir)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test


def load_mnist_dataset(args, data_dir):
    # MNIST data loaders
    trainset = datasets.MNIST(root=data_dir, train=True,
                                download=True, 
                                transform=transforms.ToTensor())
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.MNIST(root=data_dir,
                                train=False,
                                download=True, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    return loader_train, loader_test


def load_cifar_dataset(args, data_dir):
    # CIFAR-10 data loaders
    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                download=True, 
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
                                download=True, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    return loader_train, loader_test