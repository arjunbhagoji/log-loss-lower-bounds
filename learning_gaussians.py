import numpy as np
import argparse
import time
import os
import collections
import json
from scipy import special
import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment

from utils.data_utils import load_dataset_numpy

from multiprocessing import Process, Manager

def empirical_cost(X_1,X_2,eps,num_samples,return_dict):
    D_12 = scipy.spatial.distance.cdist(X_1,X_2,metric='euclidean')
    print(np.mean(D_12))
    cost_matrix = D_12 > 2*eps
    cost_matrix = cost_matrix.astype(float)
    output = linear_sum_assignment(cost_matrix)
    costs = cost_matrix[output[0], output[1]]
    cost_zero_indices = np.where(costs==0.0)
    matching_indices = (output[0][cost_zero_indices], output[1][cost_zero_indices])
    raw_cost = np.float(cost_matrix[output[0], output[1]].sum())
    mean_cost = raw_cost/(num_samples)
    min_error = (1-mean_cost)/2

    return_dict[str(num_samples)] = [min_error, mean_cost]
    
    return 

parser = argparse.ArgumentParser()
parser.add_argument('--data_dim', type=int, default=100)
parser.add_argument("--norm", default='l2',
                    help="norm to be used")
parser.add_argument('--num_samples', type=int, default=500)

args = parser.parse_args()

d=args.data_dim

random_seed = 777

evalues=np.random.default_rng(random_seed).uniform(0,1,d)

sigma = np.diag(evalues)
sigma_inv = np.linalg.inv(sigma)

mu=evalues

mean_dist=np.linalg.norm(2*mu)

delta_init = np.linspace(0,10,49)
rhs_vec=mu
w_opt=[]
eps_opt=[]
for delta in delta_init:
    lhs_mat = 0.5*(sigma+2*delta*np.eye(d))
    w=np.linalg.solve(lhs_mat,rhs_vec)
    eps_curr=delta*np.linalg.norm(w)
    print('Optimal for eps %s' % eps_curr)
    w_opt.append(w)
    eps_opt.append(eps_curr)

optimal_losses = []
for i in range(len(delta_init)):
    alpha = np.dot(w_opt[i],mu)-eps_opt[i]*np.linalg.norm(w_opt[i])
    Q_alpha = 0.5 - 0.5*special.erf(alpha/np.sqrt(2))
    optimal_loss=Q_alpha
    optimal_losses.append(optimal_loss)
    print('Optimal loss at %s is %s' % (eps_opt[i],optimal_loss))

manager = Manager()
return_dict = manager.dict()

rng = np.random.default_rng(random_seed)
sample_list = [500,5000,50000]
process_list = []
eps = 5.0
for num_samples in sample_list:
	X_1 = rng.multivariate_normal(-1.0*mu, sigma, (num_samples))
	X_2 = rng.multivariate_normal(mu, sigma, (num_samples))
	X=np.vstack((X_1,X_2))

	# empirical_loss_min, cost = empirical_cost(X_1,X_2,5.1)

	p = Process(target=empirical_cost, args=(X_1,X_2,eps,num_samples,return_dict))

	p.start()
	process_list.append(p)

for item in process_list:
    item.join()

for i in range(len(sample_list)):
	print(return_dict[str(sample_list[i])])