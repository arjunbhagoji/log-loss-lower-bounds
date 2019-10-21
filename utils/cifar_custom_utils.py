'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# def calculate_md5(fpath, chunk_size=1024 * 1024):
#     md5 = hashlib.md5()
#     with open(fpath, 'rb') as f:
#         for chunk in iter(lambda: f.read(chunk_size), b''):
#             md5.update(chunk)
#     return md5.hexdigest()


# def check_md5(fpath, md5, **kwargs):
#     return md5 == calculate_md5(fpath, **kwargs)


# def check_integrity(fpath, md5=None):
#     if not os.path.isfile(fpath):
#         return False
#     if md5 is None:
#         return True
#     return check_md5(fpath, md5)

# class StandardTransform(object):
#     def __init__(self, transform=None, target_transform=None):
#         self.transform = transform
#         self.target_transform = target_transform

#     def __call__(self, input, target):
#         if self.transform is not None:
#             input = self.transform(input)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return input, target

#     def _format_transform_repr(self, transform, head):
#         lines = transform.__repr__().splitlines()
#         return (["{}{}".format(head, lines[0])] +
#                 ["{}{}".format(" " * len(head), line) for line in lines[1:]])

#     def __repr__(self):
#         body = [self.__class__.__name__]
#         if self.transform is not None:
#             body += self._format_transform_repr(self.transform,
#                                                 "Transform: ")
#         if self.target_transform is not None:
#             body += self._format_transform_repr(self.target_transform,
#                                                 "Target transform: ")

#         return '\n'.join(body)

# class VisionDataset(data.Dataset):
#     _repr_indent = 4

#     def __init__(self, root, transforms=None, transform=None, target_transform=None):
#         if isinstance(root, torch._six.string_classes):
#             root = os.path.expanduser(root)
#         self.root = root

#         has_transforms = transforms is not None
#         has_separate_transform = transform is not None or target_transform is not None
#         if has_transforms and has_separate_transform:
#             raise ValueError("Only transforms or transform/target_transform can "
#                              "be passed as argument")

#         # for backwards-compatibility
#         self.transform = transform
#         self.target_transform = target_transform

#         if has_separate_transform:
#             transforms = StandardTransform(transform, target_transform)
#         self.transforms = transforms

#     def __getitem__(self, index):
#         raise NotImplementedError

#     def __len__(self):
#         raise NotImplementedError

#     def __repr__(self):
#         head = "Dataset " + self.__class__.__name__
#         body = ["Number of datapoints: {}".format(self.__len__())]
#         if self.root is not None:
#             body.append("Root location: {}".format(self.root))
#         body += self.extra_repr().splitlines()
#         if hasattr(self, "transforms") and self.transforms is not None:
#             body += [repr(self.transforms)]
#         lines = [head] + [" " * self._repr_indent + line for line in body]
#         return '\n'.join(lines)

#     def _format_transform_repr(self, transform, head):
#         lines = transform.__repr__().splitlines()
#         return (["{}{}".format(head, lines[0])] +
#                 ["{}{}".format(" " * len(head), line) for line in lines[1:]])

#     def extra_repr(self):
#         return ""


# def get_mean_and_std(dataset):
#     '''Compute the mean and std value of dataset.'''
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     print('==> Computing mean and std..')
#     for inputs, targets in dataloader:
#         for i in range(3):
#             mean[i] += inputs[:,i,:,:].mean()
#             std[i] += inputs[:,i,:,:].std()
#     mean.div_(len(dataset))
#     std.div_(len(dataset))
#     return mean, std

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()

# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)

#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f

class cifar10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(cifar10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.data, self.targets = self._two_c_filter()

        self._load_meta()

    def _two_c_filter(self):
        class_1 = 3
        class_2 = 7
        targets_arr = np.array(self.targets)
        c1_idx = np.where(targets_arr==class_1)
        c2_idx = np.where(targets_arr==class_2)

        X_c1 = self.data[c1_idx]
        X_c2 = self.data[c2_idx]

        curr_data = np.vstack((X_c1,X_c2))

        Y_curr = np.zeros(len(curr_data))
        Y_curr[0:len(X_c1)] = 0
        Y_curr[len(X_c1):] = 1

        curr_labels = Y_curr.tolist()
        curr_labels = [int(x) for x in curr_labels]

        return curr_data, curr_labels

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")