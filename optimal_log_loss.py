import numpy as np
import argparse
import time
import os
import collections
import json
import queue

from utils.data_utils import load_dataset_numpy

import scipy.spatial.distance

from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import maximum_flow
from utils.flow import maximum_flow


def create_graph_rep(edge_matrix,n_1,n_2):
	graph_rep = []
	for i in range(n_1+n_2+2):
	    graph_rep.append([])
	    if i==0:
	        #source
	        for j in range(n_1+n_2+2):
	            if j==0:
	                graph_rep[i].append(0)
	            elif 1<=j<=n_1:
	                graph_rep[i].append(n_2)
	            elif n_1<j<=n_1+n_2+1:
	                graph_rep[i].append(0)
	    elif 1<=i<=n_1:
	        # LHS vertices
	        for j in range(n_1+n_2+2):
	            if j<=n_1:
	                graph_rep[i].append(0)
	            elif n_1<j<=n_1+n_2:
	                if edge_matrix[i-1,j-n_1-1]:
	                    graph_rep[i].append(n_1*n_2)
	                else:
	                    graph_rep[i].append(0)
	            elif n_1+n_2<j:
	                graph_rep[i].append(0)
	    elif n_1<i<=n_1+n_2:
	        #RHS vertices
	        for j in range(n_1+n_2+2):
	            if j<=n_1+n_2:
	                graph_rep[i].append(0)
	            elif j>n_1+n_2:
	                graph_rep[i].append(n_1)
	    elif i==n_1+n_2+1:
	        #Sink
	        for j in range(n_1+n_2+2):
	            graph_rep[i].append(0)

	graph_rep_array=np.array(graph_rep)

	return graph_rep_array

def set_classifier_prob_full_flow(top_level_vertices,n_1_curr,n_2_curr):
    for item in top_level_vertices:
        if item !=0 and item != sink_idx:
            classifier_probs[item-1,0]=n_1_curr/(n_1_curr+n_2_curr)
            classifier_probs[item-1,1]=n_2_curr/(n_1_curr+n_2_curr)

def set_classifier_prob_no_flow(top_level_vertices):
    for item in top_level_vertices:
        if item !=0 and item != sink_idx:
            if item<=n_1:
                classifier_probs[item-1,0]=1
                classifier_probs[item-1,1]=0
            elif item>n_1:
                classifier_probs[item-1,0]=0
                classifier_probs[item-1,1]=1

def graph_rescale(graph_rep_curr,top_level_indices):
    n_1_curr=len(np.where(top_level_indices<=n_1)[0])-1
    n_2_curr=len(np.where(top_level_indices>n_1)[0])-1
    # source rescale
    # print(graph_rep_curr[0])
    graph_rep_curr[0,:]=graph_rep_curr[0,:]/n_2
    graph_rep_curr[0,:]*=n_2_curr
    # print(graph_rep_curr[0])
    # bipartite graph edge scale
    graph_rep_curr[1:n_1_curr+1,:]=graph_rep_curr[1:n_1_curr+1,:]/(n_1*n_2)
    graph_rep_curr[1:n_1_curr+1,:]*=(n_1_curr*n_2_curr)
    # sink edges rescale
    graph_rep_curr[n_1_curr+1:,:]=graph_rep_curr[n_1_curr+1:,:]/n_1
    graph_rep_curr[n_1_curr+1:,:]*=n_1_curr
    return graph_rep_curr,n_1_curr,n_2_curr

def find_flow_and_split(top_level_indices):
    top_level_indices_1=None
    top_level_indices_2=None
    #Create subgraph from index array provided
    graph_rep_curr = graph_rep_array[top_level_indices]
    graph_rep_curr = graph_rep_curr[:,top_level_indices]
    graph_rep_curr,n_1_curr,n_2_curr = graph_rescale(graph_rep_curr,top_level_indices)
    graph_curr=csr_matrix(graph_rep_curr)
    flow_curr = maximum_flow(graph_curr,0,len(top_level_indices)-1)
    # Checking if full flow occurred, so no need to split
    if flow_curr.flow_value==n_1_curr*n_2_curr:
        set_classifier_prob_full_flow(top_level_indices,n_1_curr,n_2_curr)
        return top_level_indices_1,top_level_indices_2, flow_curr
    elif flow_curr.flow_value==0:
        set_classifier_prob_no_flow(top_level_indices)
        return top_level_indices_1,top_level_indices_2, flow_curr
    # Finding remaining capacity edges
    edge_list_curr=flow_curr.path_edges
#     print(edge_list_curr)
    gz_idx = []
    for item in edge_list_curr:
        gz_idx.append(item[0])
        gz_idx.append(item[1])
    if len(gz_idx)>0:
        gz_idx=np.array(gz_idx)
        gz_idx_unique=np.unique(gz_idx)
        top_level_gz_idx=top_level_indices[gz_idx_unique]
        top_level_gz_idx=np.insert(top_level_gz_idx,len(top_level_gz_idx),sink_idx)
        top_level_indices_1=top_level_gz_idx
    else:
        top_level_gz_idx=np.array([0,sink_idx])
    # Indices without flow
    top_level_z_idx=np.setdiff1d(top_level_indices,top_level_gz_idx)
    if len(top_level_z_idx)>0:
        # Add source and sink back to zero flow idx array
        top_level_z_idx=np.insert(top_level_z_idx,0,0)
        top_level_z_idx=np.insert(top_level_z_idx,len(top_level_z_idx),sink_idx)
        top_level_indices_2=top_level_z_idx
    
    return top_level_indices_1,top_level_indices_2, flow_curr


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

args = parser.parse_args()

train_data, test_data, data_details = load_dataset_numpy(args, data_dir='data',
														training_time=False)
DATA_DIM = data_details['n_channels']*data_details['h_in']*data_details['w_in']

X = []
Y = []

# Pytorch normalizes tensors (so need manual here!)
if args.use_test:
	for (x,y,_, _, _) in test_data:
		X.append(x/255.)
		Y.append(y)
else:
	for (x,y,_, _, _) in train_data:
		X.append(x/255.)
		Y.append(y)

X = np.array(X)
Y = np.array(Y)

num_samples = int(len(X)/2)
print(num_samples)

class_1 = 3
class_2 = 7

if not os.path.exists('distances'):
	os.makedirs('distances')

if not os.path.exists('cost_results'):
	os.makedirs('cost_results')

if args.use_test:
	save_file_name = 'logloss_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_test_' + args.norm
else:
	save_file_name = 'logloss_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_' + args.norm

f = open('cost_results/' + save_file_name + '.txt', 'a')

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

eps = args.eps

print(eps)
# Add edge if cost 0
edge_matrix = D_12 <= 2*eps
edge_matrix = edge_matrix.astype(float)

n_1=args.num_samples
n_2=args.num_samples

# Create graph representation
graph_rep_array = create_graph_rep(edge_matrix,n_1,n_2)

q = queue.Queue()
# Initial graph indices
q.put(np.arange(n_1+n_2+2))
sink_idx=n_1+n_2+1
count=0
classifier_probs=np.zeros((n_1+n_2,2))
while not q.empty():
    print('Current queue size at eps %s is %s' % (eps,q.qsize()))
    curr_idx_list=q.get()
    # print(q.qsize())
    list_1, list_2, flow_curr=find_flow_and_split(curr_idx_list)
    # print(list_1,list_2,flow_curr.flow_value)
    if list_1 is not None:
        q.put(list_1)
    if list_2 is not None:
        q.put(list_2)

loss = 0.0
for i in range(len(classifier_probs)):
    if i<n_1:
        loss+=np.log(classifier_probs[i][0])
    elif i>=n_1:
        loss+=np.log(classifier_probs[i][1])
loss=-1*loss/len(classifier_probs)
print('Log loss for eps %s is %s' % (eps,loss))

f.write(str(eps)+','+ str(loss) + '\n')
np.savetxt('graph_data/optimal_probs/' + save_file_name + '_' + str(eps) + '.txt', classifier_probs, fmt='%.5f')
