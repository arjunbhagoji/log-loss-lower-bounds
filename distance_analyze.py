import numpy as np
import argparse
import json
import os

from utils.data_utils import load_dataset_numpy, load_dataset_tensor
from utils.io_utils import global_matching_file_name, distance_file_name, init_dirs
from utils.image_utils import save_image_simple

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

X_tr_ten, X_te_ten, data_details = load_dataset_tensor(args, data_dir='data',
                                                        training_time=False)

if args.use_test:
	train_data = False
	num_samples = int(len(X_te)/2)
else:
	train_data = True
	num_samples = int(len(X_tr)/2)

class_1 = 3
class_2 = 7

overall_dict = {}

print(global_matching_file_name(args,class_1,class_2, train_data, num_samples))
match_dict_name, match_tuple_name = global_matching_file_name(args,class_1,class_2, train_data, num_samples)
if os.path.exists(match_tuple_name):
    match_tuple = np.load(match_tuple_name)
    with open(match_dict_name, 'r') as f:
        match_dict = json.load(f)
else:
    raise ValueError('No future matching computed')

if os.path.exists(distance_file_name(args,class_1,class_2, train_data, num_samples)):
    dist_mat = np.load(distance_file_name(args, class_1, class_2, train_data, num_samples))
else:
    raise ValueError('Distances not computed')

if not os.path.exists('cost_results'):
    os.makedirs('cost_results')

if args.use_test:
    save_file_name = 'hard_rank_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_test_' + args.norm
else:
    save_file_name = 'hard_rank_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_' + args.norm

f = open('cost_results/' + save_file_name + '.txt', 'a')

figure_dir_name = 'images' + '/clean/' + args.dataset_in


count = 0
matched_count = 0
easy_list = []
hard_list = []
first_dict = {}

for k,v in match_dict.items():
    # if count == 0:
    #     curr_eps = v[1]
    #     f.write('%s:' % curr_eps)
    # if v[1] == curr_eps:
    #     f.write(k + ',')
    # else:
    #     curr_eps = v[1]
    #     f.write('\n')
    #     f.write('%s:' % curr_eps)
    #     f.write(k + ',')
    # count += 1
    # if count < 20:
    #     easy_list.append(X_tr_ten[])
    #     custom_save_image(adv_x, preds_adv, y, args, figure_dir_name, 
    #                           train_data)
    if 0 <= int(k) < 10 or num_samples <= int(k) <= num_samples + 10:
        img, label, _, _, _ = X_tr_ten[int(k)]
        first_dict[k] = [img,int(k)]

print_list = []
indices = []
for key in sorted(first_dict):
    print_list.append(first_dict[key][0])
    indices.append(first_dict[key][1])

print(len(indices))

save_image_simple(print_list, args, figure_dir_name, indices=indices)
    # if count>20:
    #     break
    # else:
    #     print(k,v)
    # count += 1

# for i in range(20):
#     # Loading matching info
#     if str(i) in match_dict:
#     	list_curr = match_dict[str(i)]
#     	matched_count += 1
#     else:
#     	list_curr = [-1, -1]
#     print(list_curr)
#     # Loading distance info
#     if i<num_samples:
#         row_idx = i
#         if str(i) in match_dict:
#             matched_dist = dist_mat[i,int(list_curr[0])-num_samples]
#             list_curr.append(matched_dist)
#         else:
#             list_curr.append(-1)
#         curr_dists = dist_mat[i,:]
#         closest_idx = np.argmin(curr_dists) + num_samples
#         closest_distance = np.min(curr_dists)
#         list_curr.extend([closest_idx, closest_distance])
#         overall_dict[str(i)] = list_curr
#     else:
#         col_idx = i % num_samples
#         if str(i) in match_dict:
#             matched_dist = dist_mat[int(list_curr[0]),col_idx]
#             list_curr.append(matched_dist)
#         else:
#             list_curr.append(-1)
#         curr_dists = dist_mat[:,col_idx]
#         closest_idx = np.argmin(curr_dists)
#         closest_distance = np.min(curr_dists)
#         list_curr.extend([closest_idx, closest_distance])
#         overall_dict[str(i)] = list_curr
#     if int(list_curr[0]) == closest_idx and str(i) in match_dict:
#         count += 1

# print(overall_dict['1'])
# print(count)
# print(matched_count)