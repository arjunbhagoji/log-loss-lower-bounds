import numpy as np
import argparse
import time
import os
import collections

from utils.data_utils import load_dataset_numpy

import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.axes_grid1 import ImageGrid

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

	plt.savefig('figures/' + save_file_name + '_eps' + str(eps) + '.png')	

def data_details():
	if 'MNIST' in args.dataset_in:
		IMAGE_ROWS = 28
		IMAGE_COLS = 28
		NUM_CHANNELS = 1
		DATA_DIM = IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS
	elif args.dataset_in == 'CIFAR-10':
		IMAGE_ROWS = 32
		IMAGE_COLS = 32
		NUM_CHANNELS = 3
		DATA_DIM = IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS
		NUM_CLASSES = 10
	elif args.dataset == 'census':
		X_train, Y_train, X_test, Y_test = data_census()
		Y_test_uncat = np.argmax(Y_test, axis=1)
		print(Y_test)
		print(Y_test_uncat)
		print('Loaded Census data')

	return DATA_DIM


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_in", default='MNIST',
                    help="dataset to be used")
parser.add_argument("--norm", default='l2',
                    help="norm to be used")
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--eps', type=float, default=None)

args = parser.parse_args()

# X_train, Y_train, X_test, Y_test = data_setup()
DATA_DIM = data_details()
train_data, test_data = load_dataset_numpy(args, data_dir='data')

X_train = []
Y_train = []
for (x,y) in train_data:
	X_train.append(x/255.)
	Y_train.append(y)
X_train = np.array(X_train)
Y_train = np.array(Y_train)

if 'MNIST' in args.dataset_in or 'CIFAR-10' in args.dataset_in:
	class_1 = 3
	class_2 = 7
	dist_mat_name = args.dataset_in + '_' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.norm + '.npy'
	X_c1 = X_train[:args.num_samples].reshape(args.num_samples, DATA_DIM)
	X_c2 = X_train[args.num_samples:].reshape(args.num_samples, DATA_DIM)
	if os.path.exists(dist_mat_name):
		D_12 = np.load(dist_mat_name)
	else:
		if args.norm == 'l2':
			D_12 = scipy.spatial.distance.cdist(X_c1,X_c2,metric='euclidean')
		elif args.norm == 'linf':
			D_12 = scipy.spatial.distance.cdist(X_c1,X_c2,metric='chebyshev')
		np.save(dist_mat_name,D_12)

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

save_file_name = str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm
save_file_name_c0 = str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm + '_cost_zero'

if not os.path.exists('cost_results'):
	os.makedirs('cost_results')

if not os.path.exists('matchings'):
	os.makedirs('matchings')

if not os.path.exists('figures'):
	os.makedirs('figures')

if not os.path.exists('degree_results'):
	os.makedirs('degree_results')

f = open('cost_results/' + save_file_name + '.txt', 'a')
f.write('eps,cost,inf_loss'+'\n')

f2 = open('degree_results/' + save_file_name + '.txt', 'a')
f2.write('eps,adn,loss_lb_loose'+'\n')

for eps in eps_list:
	print(eps)
	cost_matrix = D_12 > 2*eps
	cost_matrix = cost_matrix.astype(float)

	# Vertex degree analysis
	degrees = {}
	deg_list = []
	for i in range(2*args.num_samples):
		sample_index = i % args.num_samples
		if i<args.num_samples:
			curr_degree = np.sum(cost_matrix[sample_index,:])
			deg_list.append(curr_degree)
			degrees[str(i)] = curr_degree
		elif i>args.num_samples:
			curr_degree = np.sum(cost_matrix[:,sample_index])
			deg_list.append(curr_degree)
			degrees[str(i)] = curr_degree
	# print(len(deg_list))
	sorted_degrees = sorted(degrees.items(), key=lambda kv: kv[1])
	sorted_degrees_dict = collections.OrderedDict(sorted_degrees)
	# print(sorted_degrees_dict)

	avg_degree_norm = np.mean(deg_list)/args.num_samples
	print(avg_degree_norm)
	loss_lb_loose = (1-avg_degree_norm)/2.0
	f2.write('{:2.2},{:.4e},{:.4e}\n'.format(eps,avg_degree_norm,loss_lb_loose))

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

		print('Time taken for %s examples per class for eps %s is %s' % (args.num_samples, eps, time2-time1))

	raw_cost = np.float(cost_matrix[output[0], output[1]].sum())

	# save_adv_images(costs, output[0], output[1], X_c1, X_c2, eps)

	mean_cost = raw_cost/(args.num_samples)

	min_error = (1-mean_cost)/2

	print('At eps %s, cost: %s ; inf error: %s' % (eps, mean_cost, min_error)) 

	f.write(str(eps)+','+str(mean_cost)+','+str(min_error) + '\n')

	# Intersection analysis
	matching_indices[1] += args.num_samples
	no_matched = 2*len(matching_indices[0])
	print(no_matched)
	matching_indices = matching_indices.reshape(no_matched)
	# print(matching_indices)
	lowest_degree_indices = []
	count = 0
	for k,v in sorted_degrees_dict.items():
		if count < no_matched:
			lowest_degree_indices.append(int(k))
		count += 1
	# print(lowest_degree_indices)
	intersection = np.intersect1d(matching_indices,lowest_degree_indices)
	print(len(intersection))

f.close()
f2.close()