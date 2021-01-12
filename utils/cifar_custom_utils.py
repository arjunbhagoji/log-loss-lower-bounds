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
from .io_utils import matching_file_name, degree_file_name, distance_file_name, global_matching_file_name



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

    def __init__(self, root, args, train=True, transform=None, target_transform=None,
                 download=False, np_array=False, dropping=False, training_time=False):

        super(cifar10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.training_time = training_time
        self.train = train  # training set or test set
        self.np_array = np_array

        if training_time:
            marking_strat = args.marking_strat
        else:
            marking_strat = args.new_marking_strat

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

        self._load_meta()

        if args.n_classes == 2:
            self.data, self.targets, num_samples = self._two_c_filter(args)
        elif args.n_classes == 10:
            num_samples = len(self.data)

        self.dropping = dropping

        self.easy_idx = np.ones(2*num_samples,dtype=bool)

        self.matched_idx = -1*np.ones(2*num_samples)

        # Tracking paired points
        if marking_strat is not None and self.train:
            print('Using %s to mark' % marking_strat)
            if marking_strat == 'matched':
                mask_matched = self._matching_filter(args, num_samples)
            elif marking_strat == 'approx':
                print('Using approx filtering')
                mask_matched = self._degree_filter(args, num_samples)
            elif marking_strat == 'random':
                print('Dropping random points')
                mask_matched = self._random_filter(args, num_samples)
            elif marking_strat == 'distance':
                # Only used at test time
                mask_matched = self._distance_filter(args, num_samples)
            elif marking_strat == 'matched_future':
                mask_matched = self._matching_future_filter(args, num_samples)
        # print('No. of samples in use: {}'.format(len(self.data)))
        
        # Checking if points need to be dropped  
        if self.dropping and self.training_time and self.train:
            if marking_strat == 'distance':
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

    def _matching_filter(self, args, num_samples):
        class_1 = 3
        class_2 = 7
        mask_matched = np.ones(2*num_samples,dtype=bool)
        print(matching_file_name(args,class_1,class_2, self.train, num_samples))
        if os.path.exists(matching_file_name(args,class_1,class_2, self.train, num_samples)):
            output = np.load(matching_file_name(args, class_1, class_2, self.train, num_samples))
        else:
            raise ValueError('No matching computed')
        num_matched = len(output[0])
        if num_matched == 0:
            print('No matching')
            # return self.data, self.targets
        else:
            # Loading matching
            for i in range(num_matched):
                coin = np.random.random_sample()
                # Determining which sample to drop
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
            # Loading matching
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
        img, target = self.data[index], self.targets[index]

        easy_indc = self.easy_idx[index]

        matched_idx_curr = int(self.matched_idx[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not self.np_array:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, easy_indc, matched_idx_curr

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

    # def _load_meta(self):
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     if not check_integrity(path, self.meta['md5']):
    #         raise RuntimeError('Dataset metadata file not found or corrupted.' +
    #                            ' You can use download=True to download it')
    #     with open(path, 'rb') as infile:
    #         if sys.version_info[0] == 2:
    #             data = pickle.load(infile)
    #         else:
    #             data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target


    # def __len__(self):
    #     return len(self.data)

    # def _check_integrity(self):
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True

    # def download(self):
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    # def extra_repr(self):
    #     return "Split: {}".format("Train" if self.train is True else "Test")