import numpy as np
import argparse
import json
import os

from utils.data_utils import load_dataset_numpy
from utils.io_utils import matching_file_name, global_matching_file_name
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_in", default='MNIST',
                    help="dataset to be used")
parser.add_argument("--norm", default='l2',
                    help="norm to be used")
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--eps', type=float, default=None)
parser.add_argument('--approx_only', dest='approx_only', action='store_true')
parser.add_argument('--use_test', dest='use_test', action='store_true')
parser.add_argument('--track_hard', dest='track_hard', action='store_true')
parser.add_argument('--new_marking_strat', type=str, default=None)
parser.add_argument('--matching_path', type=str, default='matchings')

args = parser.parse_args()

X_tr, X_te, data_details = load_dataset_numpy(args, data_dir='data',
														training_time=False)

if args.use_test:
	train_data = False
	num_samples = int(len(X_te)/2)
else:
	train_data = True
	num_samples = int(len(X_tr)/2)

match_dict = OrderedDict()
class_1_indices = []
class_2_indices = []

class_1 = 3
class_2 = 7

global_dict_name, global_tuple_name = global_matching_file_name(args, class_1, 
										class_2, train_data, num_samples)

if args.dataset_in == 'MNIST':
	eps_list = np.linspace(2.2,5.0,15)
elif args.dataset_in == 'fMNIST':
	eps_list = np.linspace(2.2,6.0,20)

num_added = 0

for eps in eps_list:
	args.epsilon = eps
	print(matching_file_name(args,class_1,class_2,train_data,num_samples))
	if os.path.exists(matching_file_name(args,class_1,class_2,train_data,num_samples)):
		output = np.load(matching_file_name(args,class_1,class_2,train_data,num_samples))
	num_matched = len(output[0])
	if num_matched == 0:
		print('Nothing matched at epsilon %s' % eps)
	else:
		print(num_matched)
		for i in range(num_matched):
			class_1_idx = output[0][i]
			class_2_idx = output[1][i]
			# print(class_1_idx, class_2_idx)
			if str(class_1_idx) not in match_dict:
				num_added += 1
				match_dict[str(class_1_idx)] = [str(class_2_idx+num_samples), eps]
				class_1_indices.append(class_1_idx)
				class_2_indices.append(class_2_idx)
			if str(class_2_idx+num_samples) not in match_dict:
				num_added += 1
				c2_idx = class_2_idx+num_samples
				match_dict[str(c2_idx)] = [str(class_1_idx), eps]
				class_1_indices.append(class_1_idx)
				class_2_indices.append(class_2_idx)
			else:
				continue

matched_tuple = (class_1_indices, class_2_indices)
print(len(match_dict.keys()))
print(len(class_1_indices))
print(num_added)
np.save(global_tuple_name,matched_tuple)

with open(global_dict_name, 'w') as f:
    json.dump(match_dict, f)
