import numpy as np
import argparse
import time
import os
import collections
import json

from utils.data_utils import load_dataset_numpy

import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.axes_grid1 import ImageGrid


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


def save_adv_images(cm, indices_1, indices_2, X_c1, X_c2, eps):
	adv_indices = np.where(cm==0.0)
	X_1 = X_c1[indices_1[adv_indices]]
	X_2 = X_c2[indices_1[adv_indices]]
	adv_images = (X_1 + X_2)/2.0
	no_to_print = len(adv_images)
	# print(no_to_print)
	if 'MNIST' in args.dataset_in:
		adv_images = adv_images.reshape((no_to_print,28,28))
	elif 'CIFAR-10' in args.dataset_in:
		adv_images = adv_images.reshape((no_to_print,32,32,3))
	fig = plt.figure(1, (4., 4.))
	nrows = int(no_to_print/10.0) + 1
	# n_cols = int(no_to_print % 10.0)
	n_cols = 10
	grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(nrows, n_cols),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
                 label_mode='1')

	for i in range(no_to_print):
		if i < no_to_print:
			if 'MNIST' in args.dataset_in:
				grid[i].imshow(adv_images[i],cmap='gray')
			elif 'CIFAR-10' in args.dataset_in:
				grid[i].imshow(adv_images[i])
		# else:
		# 	grid[i].imshow(np.ones(28,28))

	plt.savefig('images/matching/' + save_file_name + '_eps' + str(eps) + '.png')


def degree_calculate(args, cost_matrix, save_file_name):
	f3_name = 'graph_data/degree_results/' + save_file_name + '_{0:.1f}'.format(eps) + '_deg_data.json'
	f4_name = 'graph_data/adj_list/' + save_file_name + '_{0:.1f}'.format(eps) + '_adj_list.json'
	# Vertex degree analysis
	if not os.path.exists(f3_name):
		f3 = open(f3_name, 'w')
		f4 = open(f4_name, 'w')
		degrees = {}
		adj_list = {}
		ind_set = set()
		ind_set_comp = set(range(2*num_samples))
		deg_list = []
		for i in range(2*num_samples):
			sample_index = i % num_samples
			if i<num_samples:
				curr_degree = np.sum(cost_matrix[sample_index,:])
				curr_neighbors = [int(i+num_samples) for i in np.where(cost_matrix[sample_index,:]==0)[0]]
				if len(curr_neighbors)>0:
					adj_list[str(i)] = [curr_neighbors, int(num_samples-curr_degree)]
				else:
					# Adding to ind set if no neighbors
					ind_set.add(i)
					ind_set_comp.remove(i)
				deg_list.append(curr_degree)
				degrees[str(i)] = curr_degree
			elif i>=num_samples:
				curr_degree = np.sum(cost_matrix[:,sample_index])
				curr_neighbors = [int(i) for i in np.where(cost_matrix[:,sample_index]==0)[0]]
				if len(curr_neighbors)>0:
					adj_list[str(i)] = [curr_neighbors, int(num_samples-curr_degree)]
				else:
					ind_set.add(i)
					ind_set_comp.remove(i)
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
	return sorted_degrees_dict, adj_list, ind_set, ind_set_comp


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

args = parser.parse_args()

train_data, test_data, data_details = load_dataset_numpy(args, data_dir='data')
DATA_DIM = data_details['n_channels']*data_details['h_in']*data_details['w_in']

X = []
Y = []

if args.use_test:
	for (x,y,_, _) in test_data:
		X.append(x/255.)
		Y.append(y)
else:
	for (x,y,_, _) in train_data:
		X.append(x/255.)
		Y.append(y)

X = np.array(X)
Y = np.array(Y)

num_samples = int(len(X)/2)
print(num_samples)

if not os.path.exists('distances'):
	os.makedirs('distances')

if 'MNIST' in args.dataset_in or 'CIFAR-10' in args.dataset_in:
	class_1 = 3
	class_2 = 7
	if args.use_test:
		dist_mat_name = args.dataset_in + '_test_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm + '.npy'
	else:
		dist_mat_name = args.dataset_in + '_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm + '.npy'
	X_c1 = X[:num_samples].reshape(num_samples, DATA_DIM)
	X_c2 = X[num_samples:].reshape(num_samples, DATA_DIM)
	if os.path.exists(dist_mat_name):
		D_12 = np.load('distances/' + dist_mat_name)
	else:
		if args.norm == 'l2':
			D_12 = scipy.spatial.distance.cdist(X_c1,X_c2,metric='euclidean')
		elif args.norm == 'linf':
			D_12 = scipy.spatial.distance.cdist(X_c1,X_c2,metric='chebyshev')
		np.save('distances/' + dist_mat_name, D_12)

if args.norm == 'l2' and 'MNIST' in args.dataset_in:
	# eps_list = np.linspace(3.2,3.8,4)
	eps_list = np.linspace(4.0,5.0,2)
	# eps_list=[2.6,2.8]
elif args.norm == 'l2' and 'CIFAR-10' in args.dataset:
	eps_list = np.linspace(4.0,10.0,13)
elif args.norm == 'linf' and 'MNIST' in args.dataset:
	eps_list = np.linspace(0.1,0.5,5)
elif args.norm == 'linf' and 'CIFAR-10' in args.dataset:
	eps_list = np.linspace(0.1,0.5,5)

if args.eps is not None:
	eps_list = [args.eps]

print(eps_list)

if args.use_test:
	save_file_name = str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_test_' + args.norm
	save_file_name_c0 = str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_test_' + args.norm + '_cost_zero'
else:
	save_file_name = str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_' + args.norm
	save_file_name_c0 = str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_' + args.norm + '_cost_zero'

if not os.path.exists('cost_results'):
	os.makedirs('cost_results')

if not os.path.exists('matchings'):
	os.makedirs('matchings')

if not os.path.exists('figures'):
	os.makedirs('figures')

if not os.path.exists('graph_data'):
	os.makedirs('graph_data')

f = open('cost_results/' + save_file_name + '.txt', 'a')
f.write('eps,cost,inf_loss'+'\n')

f2 = open('graph_data/avg_degrees/' + save_file_name + '_avg_deg' + '.txt', 'a')
f2.write('eps,adn,loss_lb_loose'+'\n')

for eps in eps_list:
	print(eps)
	cost_matrix = D_12 > 2*eps
	cost_matrix = cost_matrix.astype(float)

	# Perform all pre-computation before matching
	# sorted_degrees_dict, adj_list, ind_set, ind_set_comp = degree_calculate(args, cost_matrix, save_file_name)
	# if not os.path.exists('graph_data/greedy_ind/' + save_file_name + '_{0:.1f}.txt'.format(eps)):
	# 	ind_set = greedy_ind_set(adj_list, ind_set, ind_set_comp)
	#To-do: read in the ind set file

	# Decide if matching is to be carried out
	if not args.approx_only:
		curr_file_name = 'matchings/' + save_file_name + '_{0:.1f}.npy'.format(eps)
		curr_file_name_c0 = 'matchings/' + save_file_name_c0 + '_{0:.1f}.npy'.format(eps)
		if os.path.exists(curr_file_name):
			print('Loading computed matching')
			output = np.load(curr_file_name)
			matching_indices = np.load(curr_file_name_c0)
			costs = cost_matrix[output[0], output[1]]
		else:
			time1 = time.time()
			
			output = linear_sum_assignment(cost_matrix)
			costs = cost_matrix[output[0], output[1]]
			cost_zero_indices = np.where(costs==0.0)
			np.save(curr_file_name, output)
			
			matching_indices = (output[0][cost_zero_indices], output[1][cost_zero_indices])
			np.save(curr_file_name_c0, matching_indices)

			time2 = time.time()

			print('Time taken for %s examples per class for eps %s is %s' % (num_samples, eps, time2-time1))

		raw_cost = np.float(cost_matrix[output[0], output[1]].sum())

		# save_adv_images(costs, output[0], output[1], X_c1, X_c2, eps)

		mean_cost = raw_cost/(num_samples)

		min_error = (1-mean_cost)/2

		print('At eps %s, cost: %s ; inf error: %s' % (eps, mean_cost, min_error)) 

		f.write(str(eps)+','+str(mean_cost)+','+str(min_error) + '\n')

		# Intersection analysis
		# matching_indices[1] += num_samples
		# no_matched = 2*len(matching_indices[0])
		# print(no_matched)
		# matching_indices = matching_indices.reshape(no_matched)
		# # print(matching_indices)
		# lowest_degree_indices = []
		# count = 0
		# for k,v in sorted_degrees_dict.items():
		# 	if count < no_matched:
		# 		lowest_degree_indices.append(int(k))
		# 	count += 1
		# # print(lowest_degree_indices)
		# intersection = np.intersect1d(matching_indices,lowest_degree_indices)
		# print(len(intersection))

f.close()
f2.close()