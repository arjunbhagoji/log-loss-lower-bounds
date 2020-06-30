from __future__ import print_function
from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import json
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, verify_str_arg
from .io_utils import matching_file_name, degree_file_name, distance_file_name, global_matching_file_name


class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, args, train=True, transform=None, target_transform=None,
                 download=False, np_array=False, dropping=False, training_time=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.training_time = training_time
        self.train = train  # training set or test set
        self.np_array = np_array

        if training_time:
            self.marking_strat = args.marking_strat
        else:
            self.marking_strat = args.new_marking_strat

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        self.data, self.targets, num_samples = self._two_c_filter(args)

        self.dropping = dropping

        self.easy_idx = np.ones(2*num_samples,dtype=bool)

        self.matched_idx = -1*np.ones(2*num_samples)

        # Tracking paired points
        if self.marking_strat is not None:
            # print('Using %s to mark' % self.marking_strat)
            if self.marking_strat == 'matched':
                mask_matched = self._matching_filter(args, num_samples)
            elif self.marking_strat == 'approx':
                print('Using approx filtering')
                mask_matched = self._degree_filter(args, num_samples)
            elif self.marking_strat == 'random':
                print('Dropping random points')
                mask_matched = self._random_filter(args, num_samples)
            elif self.marking_strat == 'distance':
                # Only used at test time
                mask_matched = self._distance_filter(args, num_samples)
            elif 'matched_future' in self.marking_strat:
                mask_matched = self._matching_future_filter(args, num_samples)
        # print('No. of samples in use: {}'.format(len(self.data)))
        # Checking if points need to be dropped  
        if self.dropping and self.training_time:
            if self.marking_strat == 'distance':
                raise ValueError('Distance-based marking cannot be used at train time')
            print('Filtering training data')
            curr_data = self.data[mask_matched]
            curr_labels = np.array(self.targets)
            curr_labels = curr_labels[mask_matched]
            self.easy_idx = self.easy_idx[mask_matched]
            print('Length of data after dropping:{}'.format(len(curr_data)))    
            self.data = curr_data
            self.targets = curr_labels


    def _two_c_filter(self, args):
        class_1 = 3
        class_2 = 7
        targets_arr = np.array(self.targets)
        c1_idx = np.where(targets_arr==class_1)
        c2_idx = np.where(targets_arr==class_2)

        X_c1 = self.data[c1_idx]
        X_c2 = self.data[c2_idx]

        num_samples = args.num_samples
        if len(X_c1) < args.num_samples or len(X_c2) < args.num_samples:
            num_samples = min(len(X_c1),len(X_c2))
            # print('Culling number of samples to {}'.format(num_samples))

        if args.num_samples is not None:
            X_c1 = X_c1[:num_samples]
            X_c2 = X_c2[:num_samples]

        curr_data = np.vstack((X_c1,X_c2))
        # curr_data = X_c1.extend(X_c2)

        Y_curr = np.zeros(len(curr_data))
        Y_curr[0:len(X_c1)] = 0
        Y_curr[len(X_c1):] = 1

        curr_labels = Y_curr.tolist()
        curr_labels = [int(x) for x in curr_labels]
        
        return curr_data, curr_labels, num_samples

    def _matching_filter(self, args, num_samples):
        class_1 = 3
        class_2 = 7
        mask_matched = np.ones(2*num_samples,dtype=bool)
        # print(matching_file_name(args,class_1,class_2, self.train, num_samples))
        if os.path.exists(matching_file_name(args,class_1,class_2, self.train, num_samples)):
            output = np.load(matching_file_name(args, class_1, class_2, self.train, num_samples))
        else:
            raise ValueError('No matching computed')
        num_matched = len(output[0])
        if num_matched == 0:
            print('No matching')
            # return self.data, self.targets
        else:
            # Dropping at random
            for i in range(num_matched):
                coin = np.random.random_sample()
                if coin < 0.5:
                    mask_matched[output[0][i]] = False
                else:
                    mask_matched[num_samples+output[1][i]] = False
                # Marking hard samples
                self.easy_idx[output[0][i]] = False
                self.easy_idx[num_samples+output[1][i]] = False
                self.matched_idx[output[0][i]] = num_samples+output[1][i]
                self.matched_idx[num_samples+output[1][i]] = output[0][i]

            return mask_matched

    def _matching_future_filter(self, args, num_samples):
        class_1 = 3
        class_2 = 7
        mask_matched = np.ones(2*num_samples,dtype=bool)
        # print(global_matching_file_name(args,class_1,class_2, self.train, num_samples))
        match_dict_name, match_tuple_name = global_matching_file_name(args,class_1,class_2, self.train, num_samples)
        if os.path.exists(match_tuple_name):
            output = np.load(match_tuple_name)
            with open(match_dict_name, 'r') as f:
                output_dict = json.load(f)
        else:
            raise ValueError('No future matching computed')
        if 'replace' in self.marking_strat:
            if os.path.exists(matching_file_name(args,class_1,class_2, self.train, num_samples)):
                output_local = np.load(matching_file_name(args, class_1, class_2, self.train, num_samples))
            else:
                raise ValueError('No matching computed')
        num_matched = len(output_dict)
        # print('Marking %s using future matching' % num_matched)
        if num_matched == 0:
            print('No matching')
            # return self.data, self.targets
        else:
            # Dropping at random
            for k in output_dict:
                self.easy_idx[int(k)] = False
                self.matched_idx[int(k)] = int(output_dict[k][0])
                if 'replace' in self.marking_strat:
                    if int(k) < num_samples:
                        if int(k) in output_local[0]:
                            i = list(output_local[0]).index(int(k))
                            self.matched_idx[int(k)] = num_samples+output_local[1][i]
                            # print('%s to %s replaced at curr eps' % (int(k), num_samples+output_local[1][i]))
                    elif int(k) >= num_samples:
                        mod_k = int(k) - num_samples
                        if mod_k in output_local[1]:
                            i = list(output_local[1]).index(mod_k)
                            self.matched_idx[int(k)] = output_local[0][i]
                            # print('%s to %s replaced at curr eps' % (int(k), output_local[0][i]))

            return mask_matched

    def _degree_filter(self, args, num_samples):
        class_1 = 3
        class_2 = 7
        mask_matched = np.ones(2*num_samples,dtype=bool)
        if os.path.exists(degree_file_name(args,class_1,class_2, self.train, num_samples)):
            with open(degree_file_name(args,class_1,class_2, self.train, num_samples)) as json_file:
                degree_data = json.load(json_file)
        else:
            raise ValueError('No degree details computed')
        first_key = next(iter(degree_data))
        if degree_data[first_key] == num_samples:
            print('Only cost 1 edges present')
            # return self.data, self.targets
            return mask_matched
        else:
            # Dropping highest degree vertices from sorted dict
            count = 0
            first_time = 2*args.num_samples
            for k,v in degree_data.items():
                if v == num_samples and first_time>count:
                    first_time = count
                if count >= args.drop_thresh:
                    break
                else:
                    mask_matched[int(k)] = False
                    self.easy_idx[int(k)] = False
                count += 1
            print(len(np.where(mask_matched==False)[0]))
            print(first_time)
            return mask_matched

    def _random_filter(self, args, num_samples):
        mask_matched = np.ones(2*num_samples,dtype=bool)
        drop_indices = np.random.choice(2*num_samples, args.drop_thresh, replace=False)
        mask_matched[drop_indices] = 0
        return mask_matched
    
    def _distance_filter(self, args, num_samples):
        class_1 = 3
        class_2 = 7
        # All points are matched to the closest point
        mask_matched = np.ones(2*num_samples,dtype=bool)
        # print(matching_file_name(args,class_1,class_2, self.train, num_samples))
        if os.path.exists(distance_file_name(args,class_1,class_2, self.train, num_samples)):
            dist_mat = np.load(distance_file_name(args, class_1, class_2, self.train, num_samples))
        else:
            raise ValueError('Distances not computed')
        # Pairing based on distance
        for i in range(2*num_samples):
            if i<num_samples:
                row_idx = i
                curr_dists = dist_mat[row_idx,:]
                closest_idx = np.argmin(curr_dists)
                self.matched_idx[i] = num_samples+closest_idx
                self.easy_idx[i] = False
            else:
                col_idx = i % num_samples
                curr_dists = dist_mat[:,col_idx]
                closest_idx = np.argmin(curr_dists)
                self.matched_idx[i] = closest_idx
                self.easy_idx[i] = False

        return mask_matched

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        
        easy_indc = self.easy_idx[index]

        matched_idx_curr = int(self.matched_idx[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not self.np_array:
            img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, easy_indc, matched_idx_curr

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")




class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
            and  ``Fashion-MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
         "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# class KMNIST(MNIST):
#     """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset where ``KMNIST/processed/training.pt``
#             and  ``KMNIST/processed/test.pt`` exist.
#         train (bool, optional): If True, creates dataset from ``training.pt``,
#             otherwise from ``test.pt``.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """
#     urls = [
#         'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
#         'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
#         'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
#         'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz',
#     ]
#     classes = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']


# class EMNIST(MNIST):
#     """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
#             and  ``EMNIST/processed/test.pt`` exist.
#         split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
#             ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
#             which one to use.
#         train (bool, optional): If True, creates dataset from ``training.pt``,
#             otherwise from ``test.pt``.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """
#     # Updated URL from https://www.nist.gov/node/1298471/emnist-dataset since the
#     # _official_ download link
#     # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download
#     # is (currently) unavailable
#     url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
#     splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

#     def __init__(self, root, split, **kwargs):
#         self.split = verify_str_arg(split, "split", self.splits)
#         self.training_file = self._training_file(split)
#         self.test_file = self._test_file(split)
#         super(EMNIST, self).__init__(root, **kwargs)

#     @staticmethod
#     def _training_file(split):
#         return 'training_{}.pt'.format(split)

#     @staticmethod
#     def _test_file(split):
#         return 'test_{}.pt'.format(split)

#     def download(self):
#         """Download the EMNIST data if it doesn't exist in processed_folder already."""
#         import shutil

#         if self._check_exists():
#             return

#         makedir_exist_ok(self.raw_folder)
#         makedir_exist_ok(self.processed_folder)

#         # download files
#         print('Downloading and extracting zip archive')
#         download_and_extract_archive(self.url, download_root=self.raw_folder, filename="emnist.zip",
#                                      remove_finished=True)
#         gzip_folder = os.path.join(self.raw_folder, 'gzip')
#         for gzip_file in os.listdir(gzip_folder):
#             if gzip_file.endswith('.gz'):
#                 extract_archive(os.path.join(gzip_folder, gzip_file), gzip_folder)

#         # process and save as torch files
#         for split in self.splits:
#             print('Processing ' + split)
#             training_set = (
#                 read_image_file(os.path.join(gzip_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
#                 read_label_file(os.path.join(gzip_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
#             )
#             test_set = (
#                 read_image_file(os.path.join(gzip_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
#                 read_label_file(os.path.join(gzip_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
#             )
#             with open(os.path.join(self.processed_folder, self._training_file(split)), 'wb') as f:
#                 torch.save(training_set, f)
#             with open(os.path.join(self.processed_folder, self._test_file(split)), 'wb') as f:
#                 torch.save(test_set, f)
#         shutil.rmtree(gzip_folder)

#         print('Done!')


# class QMNIST(MNIST):
#     """`QMNIST <https://github.com/facebookresearch/qmnist>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset whose ``processed''
#             subdir contains torch binary files with the datasets.
#         what (string,optional): Can be 'train', 'test', 'test10k',
#             'test50k', or 'nist' for respectively the mnist compatible
#             training set, the 60k qmnist testing set, the 10k qmnist
#             examples that match the mnist testing set, the 50k
#             remaining qmnist testing examples, or all the nist
#             digits. The default is to select 'train' or 'test'
#             according to the compatibility argument 'train'.
#         compat (bool,optional): A boolean that says whether the target
#             for each example is class number (for compatibility with
#             the MNIST dataloader) or a torch vector containing the
#             full qmnist information. Default=True.
#         download (bool, optional): If true, downloads the dataset from
#             the internet and puts it in root directory. If dataset is
#             already downloaded, it is not downloaded again.
#         transform (callable, optional): A function/transform that
#             takes in an PIL image and returns a transformed
#             version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform
#             that takes in the target and transforms it.
#         train (bool,optional,compatibility): When argument 'what' is
#             not specified, this boolean decides whether to load the
#             training set ot the testing set.  Default: True.

#     """

#     subsets = {
#         'train': 'train',
#         'test': 'test', 'test10k': 'test', 'test50k': 'test',
#         'nist': 'nist'
#     }
#     urls = {
#         'train': ['https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz',
#                   'https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz'],
#         'test': ['https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz',
#                  'https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz'],
#         'nist': ['https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-images-idx3-ubyte.xz',
#                  'https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-labels-idx2-int.xz']
#     }
#     classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
#                '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

#     def __init__(self, root, what=None, compat=True, train=True, **kwargs):
#         if what is None:
#             what = 'train' if train else 'test'
#         self.what = verify_str_arg(what, "what", tuple(self.subsets.keys()))
#         self.compat = compat
#         self.data_file = what + '.pt'
#         self.training_file = self.data_file
#         self.test_file = self.data_file
#         super(QMNIST, self).__init__(root, train, **kwargs)

#     def download(self):
#         """Download the QMNIST data if it doesn't exist in processed_folder already.
#            Note that we only download what has been asked for (argument 'what').
#         """
#         if self._check_exists():
#             return
#         makedir_exist_ok(self.raw_folder)
#         makedir_exist_ok(self.processed_folder)
#         urls = self.urls[self.subsets[self.what]]
#         files = []

#         # download data files if not already there
#         for url in urls:
#             filename = url.rpartition('/')[2]
#             file_path = os.path.join(self.raw_folder, filename)
#             if not os.path.isfile(file_path):
#                 download_url(url, root=self.raw_folder, filename=filename, md5=None)
#             files.append(file_path)

#         # process and save as torch files
#         print('Processing...')
#         data = read_sn3_pascalvincent_tensor(files[0])
#         assert(data.dtype == torch.uint8)
#         assert(data.ndimension() == 3)
#         targets = read_sn3_pascalvincent_tensor(files[1]).long()
#         assert(targets.ndimension() == 2)
#         if self.what == 'test10k':
#             data = data[0:10000, :, :].clone()
#             targets = targets[0:10000, :].clone()
#         if self.what == 'test50k':
#             data = data[10000:, :, :].clone()
#             targets = targets[10000:, :].clone()
#         with open(os.path.join(self.processed_folder, self.data_file), 'wb') as f:
#             torch.save((data, targets), f)

#     def __getitem__(self, index):
#         # redefined to handle the compat flag
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img.numpy(), mode='L')
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.compat:
#             target = int(target[0])
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target

#     def extra_repr(self):
#         return "Split: {}".format(self.what)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x
