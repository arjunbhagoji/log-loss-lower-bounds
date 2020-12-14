import numpy as np
import argparse
import time
import os
import collections
import json

import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment


def two_class_filter(X, Y, class_1, class_2):
    c1_idx = np.where(Y==class_1)
    c2_idx = np.where(Y==class_2)

    X_c1 = X[c1_idx]
    X_c2 = X[c2_idx]

    num_samples = min(len(X_c1),len(X_c2))
    print('No. of samples is %s' % num_samples)

    X_c1 = X_c1[:num_samples]
    X_c2 = X_c2[:num_samples]

    return X_c1, X_c2


def greedy_ind_set(adj_list, ind_set, graph_track):
	f5 = open('graph_data/greedy_ind/' + save_file_name + '_{0:.1f}.txt'.format(eps), 'w')
	sorted_adj_list = sorted(adj_list.items(), key=lambda kv: kv[1][1])
	# print(sorted_adj_list)
	# sorted_adj_list = collections.OrderedDict(sorted_adj_list)
	while len(graph_track)>0:
		# print(sorted_adj_list)
		# print(graph_track, ind_set)
		curr_vertex_data = sorted_adj_list[0]
		# print(int(curr_vertex_data[0]))
		ind_set.add(int(curr_vertex_data[0]))
		if int(curr_vertex_data[0]) in graph_track: 
			graph_track.remove(int(curr_vertex_data[0]))
		for i in curr_vertex_data[1][0]:
			if i in graph_track:
				graph_track.remove(i)
		for item in sorted_adj_list:
			if int(item[0]) in curr_vertex_data[1][0]:
				sorted_adj_list.remove(item)
				# print(item)
		sorted_adj_list = sorted_adj_list[1:]
	print(len(ind_set))
	for item in ind_set:
		f5.write(str(item)+',')
	return ind_set

def degree_calculate(args, cost_matrix, save_file_name):
	f3_name = 'graph_data/degree_results/' + save_file_name + '_{0:.1f}'.format(eps) + '_deg_data.json'
	f4_name = 'graph_data/adj_list/' + save_file_name + '_{0:.1f}'.format(eps) + '_adj_list.json'
	# Vertex degree analysis
	if not os.path.exists(f3_name):
		f3 = open(f3_name, 'w')
		f4 = open(f4_name, 'w')
		degrees = {}
		adj_list = {}
		# ind_set = set()
		# ind_set_comp = set(range(2*num_samples))
		deg_list = []
		for i in range(2*num_samples):
			sample_index = i % num_samples
			if i<num_samples:
				curr_degree = np.sum(cost_matrix[sample_index,:])
				curr_neighbors = [int(i+num_samples) for i in np.where(cost_matrix[sample_index,:]==0)[0]]
				if len(curr_neighbors)>0:
					adj_list[str(i)] = [curr_neighbors, int(num_samples-curr_degree)]
				# else:
				# 	# Adding to ind set if no neighbors
				# 	ind_set.add(i)
				# 	ind_set_comp.remove(i)
				deg_list.append(curr_degree)
				degrees[str(i)] = curr_degree
			elif i>=num_samples:
				curr_degree = np.sum(cost_matrix[:,sample_index])
				curr_neighbors = [int(i) for i in np.where(cost_matrix[:,sample_index]==0)[0]]
				if len(curr_neighbors)>0:
					adj_list[str(i)] = [curr_neighbors, int(num_samples-curr_degree)]
				# else:
				# 	ind_set.add(i)
				# 	ind_set_comp.remove(i)
				deg_list.append(curr_degree)
				degrees[str(i)] = curr_degree
		sorted_degrees = sorted(degrees.items(), key=lambda kv: kv[1])
		sorted_degrees_dict = collections.OrderedDict(sorted_degrees)
		# print(sorted_degrees_dict)
		# print(adj_list)
		# print(ind_set, ind_set_comp)
		json.dump(sorted_degrees_dict, f3)
		json.dump(adj_list, f4)
		avg_degree_norm = np.mean(deg_list)/num_samples
		loss_lb_loose = (1-avg_degree_norm)/2.0
		f2.write('{:2.2},{:.4e},{:.4e}\n'.format(eps,avg_degree_norm,loss_lb_loose))
		f3.close()
		f4.close()
	else:
		with open(f3_name) as json_file1:
			sorted_degrees_dict = json.load(json_file1)
		with open(f4_name) as json_file2:
			adj_list = json.load(json_file2)
	return sorted_degrees_dict, adj_list


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_in", type=str, default='cifar10_dim224_basic',
                    help="dataset to be used", choices=["flowers_basic", 
                    "caltech101_basic", "stcars_basic", "fgvcplanes_basic", 
                    "cifar10_dim224_basic", "cifar100_dim224_basic"])
parser.add_argument("--net", type=str, default='', choices=["", "_1x_infomin", 
					"_1x_mocov2",  "_1x_simclr",  "_1x_pirl", "_1x_cmc"])
parser.add_argument("--norm", default='l2',
                    help="norm to be used")
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--eps', type=float, default=None)
parser.add_argument('--approx_only', dest='approx_only', action='store_true')
parser.add_argument('--use_test', dest='use_test', action='store_true')
parser.add_argument('--track_hard', dest='track_hard', action='store_true')
parser.add_argument('--new_marking_strat', type=str, default=None)
parser.add_argument('--no_budgets', type=int, default=1)

args = parser.parse_args()

num_classes = 10

data_path = '/data/nvme/vvikash/datasets/hidden_features'

base_model = 'ResNet50{}_{}'.format(args.net, args.dataset_in)

X_train = np.load(data_path+'/'+base_model+'/shard_0/features_train.npy')
X_test = np.load(data_path+'/'+base_model+'/shard_0/features_test.npy')
Y_train = np.load(data_path+'/'+base_model+'/shard_0/labels_train.npy')
Y_test = np.load(data_path+'/'+base_model+'/shard_0/labels_test.npy')

# Make paths
if not os.path.exists('feature_distances'):
	os.makedirs('feature_distances')

if not os.path.exists('cost_results'):
	os.makedirs('cost_results')

if not os.path.exists('matchings/' + args.dataset_in):
	os.makedirs('matchings/' + args.dataset_in)

if not os.path.exists('figures'):
	os.makedirs('figures')

if not os.path.exists('graph_data'):
	os.makedirs('graph_data')

if args.use_test:
	save_file_name = args.dataset_in + args.net + '_test_' + '_' + args.norm
	save_file_name_c0 = args.dataset_in + args.net + '_test_' + '_' + args.norm + '_cost_zero'
else:
	save_file_name = args.dataset_in + args.net + '_' + args.norm
	save_file_name_c0 = args.dataset_in + args.net + '_' + args.norm + '_cost_zero'

f = open('cost_results/' + save_file_name + '.txt', 'a')
f.write('eps,costs'+'\n')

eps_list = np.linspace(0.0,3.0,16)

# if args.eps is not None:
# 	if args.no_budgets == 1:
# 		eps_list = [2*args.eps]
# 	elif args.no_budgets == 2:
# 		eps_1 = 0.0
# 		eps_2 = args.eps
# 		eps_list = [2*eps_1, eps_1+eps_2, eps_1+eps_2, 2*eps_2]

print(eps_list)

time_start = time.time()

for i, eps in enumerate(eps_list):
	print('Eps: %s' % eps)
	cost_list = []
	class_1 = 0
	class_2 = 1
	while class_1<num_classes and class_2<num_classes:
		print(class_1, class_2)

		if args.use_test:
			X_1, X_2 = two_class_filter(X_test, Y_test, class_1, class_2)
		else:
			X_1, X_2 = two_class_filter(X_train, Y_train, class_1, class_2)

		assert len(X_1) == len(X_2)

		num_samples = len(X_1)

		if args.use_test:
			dist_mat_name = args.dataset_in + args.net + '_test_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm + '.npy'
		else:
			dist_mat_name = args.dataset_in + args.net + '_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm + '.npy'

		if os.path.exists(dist_mat_name):
			D_12 = np.load('feature_distances/' + dist_mat_name)
		else:
			if args.norm == 'l2':
				D_12 = scipy.spatial.distance.cdist(X_1,X_2,metric='euclidean')
			elif args.norm == 'linf':
				D_12 = scipy.spatial.distance.cdist(X_1,X_2,metric='chebyshev')
			np.save('feature_distances/' + dist_mat_name, D_12)
		print('Max distance is %s' % np.amax(D_12))
		print('Min distance is %s' % np.amin(D_12))

		f2 = open('graph_data/avg_degrees/' + save_file_name + '_avg_deg' + '.txt', 'a')
		f2.write('eps,adn'+'\n')

		# cost_matrix = np.zeros((args.no_budgets*num_samples, args.no_budgets*num_samples))

		print(eps)
		cost_matrix = D_12 > 2*eps
		# print('Number of cost 1 edges: %s' % cost_matrix.sum())
		cost_matrix = cost_matrix.astype(float)
		# print(cost_matrix)

		# Perform all pre-computation before matching
		# sorted_degrees_dict, adj_list = degree_calculate(args, cost_matrix, save_file_name)
		# if not os.path.exists('graph_data/greedy_ind/' + save_file_name + '_{0:.1f}.txt'.format(eps)):
		# 	ind_set = greedy_ind_set(adj_list, ind_set, ind_set_comp)
		#To-do: read in the ind set file

		# Decide if matching is to be carried out
		if not args.approx_only:
			if args.no_budgets==1:
				curr_file_name = 'matchings/' + args.dataset_in + '/' + save_file_name + '_' + str(class_1) + '_' + str(class_2) + '_{0:.1f}.npy'.format(eps)
				curr_file_name_c0 = 'matchings/' + args.dataset_in + '/' + save_file_name_c0 + '_' + str(class_1) + '_' + str(class_2) + '_{0:.1f}.npy'.format(eps)
			# elif args.no_budgets==2:
			# 	curr_file_name = 'matchings/' + args.dataset_in + '/' + save_file_name + '_{0:.1f}_{0:.1f}.npy'.format(0.0,args.eps)
			# 	curr_file_name_c0 = 'matchings/' + args.dataset_in + '/' + save_file_name_c0 + '_{0:.1f}_{0:.1f}.npy'.format(0.0,args.eps)
			if os.path.exists(curr_file_name):
				print('Loading computed matching')
				output = np.load(curr_file_name)
				print('Matching: %s' % output)
				matching_indices = np.load(curr_file_name_c0)
				costs = cost_matrix[output[0], output[1]]
				print(costs)
			else:
				time1 = time.time()
				
				output = linear_sum_assignment(cost_matrix)

				# Gives 1d array of relevant costs
				costs = cost_matrix[output[0], output[1]]

				cost_zero_indices = np.where(costs==0.0)
				np.save(curr_file_name, output)
				
				matching_indices = (output[0][cost_zero_indices], output[1][cost_zero_indices])
				np.save(curr_file_name_c0, matching_indices)

				time2 = time.time()

				print('Time taken for %s examples per class for eps %s is %s' % (num_samples, eps, time2-time1))

			raw_cost = np.float(cost_matrix[output[0], output[1]].sum())
			print('Raw cost: %s' % raw_cost)

			# save_adv_images(costs, output[0], output[1], X_c1, X_c2, eps)

			mean_cost = raw_cost/(args.no_budgets*num_samples)

			min_error = (1-mean_cost)/2

			print('At eps %s, cost: %s ; inf error: %s' % (eps, mean_cost, min_error)) 

			cost_list.append(mean_cost)

			class_2 += 1
			if class_2==num_classes:
				class_1 += 1
				class_2 = class_1 + 1

	f.write(str(eps))
	for item in cost_list:
		f.write(','+str(item))
	f.write('\n')

f.close()
f2.close()

time_end = time.time()
print('Time taken for %s is %s' % (save_file_name, time_end-time_start))