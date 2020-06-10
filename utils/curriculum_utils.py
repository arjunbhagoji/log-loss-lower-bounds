import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .mnist_custom_utils import MNIST, FashionMNIST
from .io_utils import global_matching_file_name

import os
import numpy as np
import json

class_1 = 3
class_2 = 7

def curriculum_setter(args, data_dir):
    print('Loading data for curriculum')
    loader_list = []
    training_time = True
    trainset_all = MNIST(root=data_dir, args=args, train=True,
                        download=False, transform=transforms.ToTensor(),
                        training_time=training_time, dropping=False)
    loader_train_all = torch.utils.data.DataLoader(trainset_all, 
                            batch_size=args.batch_size,
                            shuffle=True)
    if 'local_match' in args.curriculum:
        easy_list = []
        for item in trainset_all:
            if item[3]:
            	easy_list.append(item)
        print(len(easy_list))
        loader_train_easy = torch.utils.data.DataLoader(easy_list, 
                                batch_size=args.batch_size,
                                shuffle=True)
        loader_list.append(loader_train_easy)
        loader_list.append(loader_train_all)
    elif 'global_match' in args.curriculum:
        samples_per_set = int(2*args.num_samples/args.num_sets)
        assert 2*args.num_samples % args.num_sets==0
        print(global_matching_file_name(args,class_1,class_2, True, args.num_samples))
        match_dict_name, _ = global_matching_file_name(args,class_1,class_2, True, args.num_samples)
        if os.path.exists(match_dict_name):
            with open(match_dict_name, 'r') as f:
                output_dict = json.load(f)
        all_indices = set(range(2*args.num_samples))
        # curriculum_matrix = np.zeros((args.num_sets,samples_per_set))
        curriculum_matrix = []
        for i in range(args.num_sets):
            curriculum_matrix.append([])
        # Hardest data is loaded into last list
        row_idx = -1
        col_idx = 0
        for k in output_dict:
            curriculum_matrix[row_idx].append(trainset_all[int(k)])
            # print(output_dict[k])
            all_indices.discard(int(k))
            col_idx += 1
            if col_idx==samples_per_set:
                row_idx-=1
                col_idx=0
        print(row_idx,col_idx)
        print(len(all_indices))
        if len(all_indices)>0:
            for item in all_indices:
                curriculum_matrix[row_idx].append(trainset_all[int(k)])
                col_idx += 1
                if col_idx==samples_per_set:
                    row_idx-=1
                    col_idx=0
        print(row_idx,col_idx)
        print(len(loader_list))

        # Adding easiest data first
        for i in range(args.num_sets):
            curr_data = []
            for j in range(i+1):
                curr_data.extend(curriculum_matrix[i])
            curr_loader_train = torch.utils.data.DataLoader(curr_data, 
                    batch_size=args.batch_size,
                    shuffle=True)
            loader_list.append(curr_loader_train)
            print(len(curr_data))

    return loader_list

def curriculum_checker(args, epoch, curriculum_step, early_stop_counter):
    # if early_stop_counter>5:
    #     curriculum_step += 1
    #     print('Shifting to curriculum step %s at epoch %s' % (curriculum_step,epoch))
    #     early_stop_counter = 0
    if epoch >= (curriculum_step+1)*(args.train_epochs/args.num_sets):
        curriculum_step += 1
        print('Shifting to curriculum step %s at epoch %s' % (curriculum_step,epoch))
    return curriculum_step, early_stop_counter