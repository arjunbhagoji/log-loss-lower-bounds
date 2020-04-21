import numpy as np
import argparse
import json
import os

from utils.data_utils import load_dataset_numpy
from utils.io_utils import matching_file_name, global_matching_file_name


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

match_dict = {}
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
			if class_1_idx in class_1_indices:
				# print('Skipping %s and %s' % (class_1_idx, class_2_idx))
				continue
			else:
				match_dict[str(class_1_idx)] = [str(class_2_idx+num_samples), str(eps)]
				match_dict[str(class_2_idx+num_samples)] = [str(class_1_idx), str(eps)]
				class_1_indices.append(class_1_idx)
				class_2_indices.append(class_2_idx)

matched_tuple = (class_1_indices, class_2_indices)
print(len(class_1_indices))
np.save(global_tuple_name,matched_tuple)

with open(global_dict_name, 'w') as f:
    json.dump(match_dict, f)
