import numpy as np
from scipy import special
from scipy.stats import multivariate_normal

from scipy import integrate
import scipy.spatial.distance
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import maximum_flow
from utils.flow import _make_edge_pointers

import queue
import pickle
import time
import os


def find_remaining_cap_edges(edge_ptr,capacities,heads,tails, source, sink):
    ITYPE = np.int32
    n_verts = edge_ptr.shape[0] - 1
    n_edges = capacities.shape[0]
    ITYPE_MAX = np.iinfo(ITYPE).max

    # Our result array will keep track of the flow along each edge
    flow = np.zeros(n_edges, dtype=ITYPE)

    # Create a circular queue for breadth-first search. Elements are
    # popped dequeued at index start and queued at index end.
    q = np.empty(n_verts, dtype=ITYPE)

    # Create an array indexing predecessor edges
    pred_edge = np.empty(n_verts, dtype=ITYPE)

    # While augmenting paths from source to sink exist
    for k in range(n_verts):
        pred_edge[k] = -1
    path_edges = []
    # Reset queue to consist only of source
    q[0] = source
    start = 0
    end = 1
    # While we have not found a path, and queue is not empty
    path_found = False
    while start != end and not path_found:
        # Pop queue
        cur = q[start]
        start += 1
        if start == n_verts:
            start = 0
        # Loop over all edges from the current vertex
        for e in range(edge_ptr[cur], edge_ptr[cur + 1]):
            t = heads[e]
            if pred_edge[t] == -1 and t != source and\
                    capacities[e] > flow[e]:
                pred_edge[t] = e
                path_edges.append((cur,t))
                if t == sink:
                    path_found = True
                    break
                # Push to queue
                q[end] = t
                end += 1
                if end == n_verts:
                    end = 0
    return path_edges

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

def set_classifier_prob_full_flow(top_level_vertices,n_1_curr,n_2_curr,sink_idx,classifier_probs):
    for item in top_level_vertices:
        if item !=0 and item != sink_idx:
            classifier_probs[item-1,0]=n_1_curr/(n_1_curr+n_2_curr)
            classifier_probs[item-1,1]=n_2_curr/(n_1_curr+n_2_curr)

def set_classifier_prob_no_flow(top_level_vertices,sink_idx,n_1,classifier_probs):
    for item in top_level_vertices:
        if item !=0 and item != sink_idx:
            if item<=n_1:
                classifier_probs[item-1,0]=1
                classifier_probs[item-1,1]=0
            elif item>n_1:
                classifier_probs[item-1,0]=0
                classifier_probs[item-1,1]=1

def graph_rescale(graph_rep_curr,top_level_indices,n_1,n_2):
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


def find_flow_and_split(top_level_indices,graph_rep_array,n_1,n_2,classifier_probs):
    sink_idx=n_1+n_2+1
    top_level_indices_1=None
    top_level_indices_2=None
    #Create subgraph from index array provided
    graph_rep_curr = graph_rep_array[top_level_indices]
    graph_rep_curr = graph_rep_curr[:,top_level_indices]
    graph_rep_curr,n_1_curr,n_2_curr = graph_rescale(graph_rep_curr,top_level_indices,n_1,n_2)
    graph_curr=csr_matrix(graph_rep_curr)
    flow_curr = maximum_flow(graph_curr,0,len(top_level_indices)-1)
    # Checking if full flow occurred, so no need to split
    if flow_curr.flow_value==n_1_curr*n_2_curr:
        set_classifier_prob_full_flow(top_level_indices,n_1_curr,n_2_curr, sink_idx,classifier_probs)
        return top_level_indices_1,top_level_indices_2, flow_curr
    elif flow_curr.flow_value==0:
        set_classifier_prob_no_flow(top_level_indices,sink_idx,n_1,classifier_probs)
        return top_level_indices_1,top_level_indices_2, flow_curr
    # Finding remaining capacity edges
    remainder_array = graph_curr-flow_curr.residual

    rev_edge_ptr, tails = _make_edge_pointers(remainder_array)

    edge_ptr=remainder_array.indptr
    capacities=remainder_array.data
    heads=remainder_array.indices

    edge_list_curr = find_remaining_cap_edges(edge_ptr,capacities,heads,tails,0,len(top_level_indices)-1)

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

def log_empirical_cost(X_1,X_2,eps):
    D_12 = scipy.spatial.distance.cdist(X_1,X_2,metric='euclidean')
    edge_matrix = D_12 <= 2*eps
    edge_matrix = edge_matrix.astype(float)

    num_edges = len(np.where(edge_matrix!=0)[0])
    # num_edges_list.append(num_edges)

    n_1=len(X_1)
    n_2=len(X_2)
    print(n_1,n_2)

    # Create graph representation
    graph_rep_array = create_graph_rep(edge_matrix,n_1,n_2)

    time1= time.clock()
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
        list_1, list_2, flow_curr=find_flow_and_split(curr_idx_list,graph_rep_array,n_1,n_2,classifier_probs)
        # print(list_1,list_2,flow_curr.flow_value)
        if list_1 is not None:
            q.put(list_1)
        if list_2 is not None:
            q.put(list_2)
    time2 = time.clock()
    
    loss = 0.0
    for i in range(len(classifier_probs)):
        if i<n_1:
            loss+=np.log(classifier_probs[i][0])
        elif i>=n_1:
            loss+=np.log(classifier_probs[i][1])
    loss=-1*loss/len(classifier_probs)
    
    return classifier_probs, loss

d_list = [2,10]
sample_num_list = [1000,5000]

if os.path.exists("gauss_sample.dat"):
    PIK1 = "gauss_sample.dat"
    with open(PIK1, "rb") as f:
        sample_list=pickle.load(f)
    PIK2 = "gauss_params.dat"
    with open(PIK2, "rb") as f:
        param_list=pickle.load(f)
    PIK3 = "eps_opt.dat"
    with open(PIK3, "rb") as f:
        eps_opt_list=pickle.load(f)
    PIK4 = "opt_01_losses.dat"
    with open(PIK4, "rb") as f:
        optimal_loss_list=pickle.load(f)
    PIK5 = "emp_log_losses.dat"
    with open(PIK5, "rb") as f:
        emp_log_loss_list=pickle.load(f)
    PIK6 = "opt_w.dat"
    with open(PIK6, "rb") as f:
        w_opt_list=pickle.load(f)
    PIK7 = "gauss_sample_test.dat"
    with open(PIK7, "rb") as f:
        sample_test_list=pickle.load(f)
else:
    rng = np.random.default_rng(77)
    param_list = []
    sample_list = []
    sample_list_test = []
    emp_log_loss_list = []
    eps_opt_list = []
    optimal_loss_list = []
    w_opt_list=[]
    for i,d in enumerate(d_list):
        print(i,d)
        # Gaussian params
        evalues=np.random.default_rng().uniform(0,1,d)
        sigma = np.diag(evalues)
        sigma_inv = np.linalg.inv(sigma)
        mu=(alpha_star)*np.sqrt(evalues)/np.sqrt(d)
        param_list.append([mu,evalues])
        # Run eps search
        delta_init = np.linspace(0,3,10)
        rhs_vec=mu
        w_opt=[]
        eps_opt=[]
        for delta in delta_init:
            lhs_mat = 0.5*(sigma+2*delta*np.eye(d))
            w=np.linalg.solve(lhs_mat,rhs_vec)
            eps_curr=delta*np.linalg.norm(w)
            #print('Optimal for eps %s' % eps_curr)
            w_opt.append(w)
            eps_opt.append(eps_curr)
        eps_opt_list.append(eps_opt)
        w_opt_list.append(w_opt)
        # Compute optimal loss
        optimal_losses = []
        for i in range(len(delta_init)):
            alpha = (-eps_opt[i]*np.linalg.norm(w_opt[i])+np.dot(w_opt[i],mu))/np.sqrt(np.dot(w_opt[i],np.dot(sigma,w_opt[i])))
            Q_alpha = 0.5 - 0.5*special.erf(alpha/np.sqrt(2))
            optimal_loss=Q_alpha
            optimal_losses.append(optimal_loss)
            #print('Optimal loss at %s is %s' % (eps_opt[i],optimal_loss))
        optimal_loss_list.append(optimal_losses)
        # Generate samples
        samples=[]
        emp_loss=[]
        for j, num_samples in enumerate(sample_num_list):
            print(j,num_samples)
            X_1 = rng.multivariate_normal(-1.0*mu, sigma, (num_samples))
            X_2 = rng.multivariate_normal(mu, sigma, (num_samples))
            X=np.vstack((X_1,X_2))
            samples.append(X)
            emp_loss_per_sample=[]
            for eps in eps_opt:
                classifier_probs, log_loss_min = log_empirical_cost(X_1,X_2,eps)
                emp_loss_per_sample.append(log_loss_min)
            emp_loss.append(emp_loss_per_sample)
        emp_log_loss_list.append(emp_loss)
        sample_list.append(samples)
        # Generate test samples
        num_samples_test=1000
        X_1_test = rng.multivariate_normal(-1.0*mu, sigma, (num_samples_test))
        X_2_test = rng.multivariate_normal(mu, sigma, (num_samples_test))
        X_test=np.vstack((X_1_test,X_2_test))
        sample_list_test.append(X_test)
        print(optimal_losses)
        print(emp_loss)
    PIK1 = "gauss_sample.dat"
    with open(PIK1, "wb") as f:
        pickle.dump(sample_list, f)
    PIK2 = "gauss_params.dat"
    with open(PIK2, "wb") as f:
        pickle.dump(param_list, f)
    PIK3 = "eps_opt.dat"
    with open(PIK3, "wb") as f:
        pickle.dump(eps_opt_list, f)
    PIK4 = "opt_01_losses.dat"
    with open(PIK4, "wb") as f:
        pickle.dump(optimal_loss_list, f)
    PIK5 = "emp_log_losses.dat"
    with open(PIK5, "wb") as f:
        pickle.dump(emp_log_loss_list, f)
    PIK6 = "opt_w.dat"
    with open(PIK6, "wb") as f:
        pickle.dump(w_opt_list, f)
    PIK7 = "gauss_sample_test.dat"
    with open(PIK7, "wb") as f:
        pickle.dump(sample_list_test, f)

# High epsilon runs
eps_list=np.linspace(1.5,3,4)
high_eps_loss_list=[]
for i,d in enumerate(d_list):
    samples=[]
    emp_loss=[]
    for j, num_samples in enumerate(sample_num_list):
        print(d,num_samples)
        high_eps_loss_list.append([])
        samples=sample_list[i][j]
        X_1=samples[:num_samples]
        X_2=samples[num_samples:]
        emp_loss_per_sample=[]
        for eps in eps_list:
            classifier_probs, log_loss_min = log_empirical_cost(X_1,X_2,eps)
            print(log_loss_min)
            high_eps_loss_list[j].append(log_loss_min)
            
# Computing the optimal log loss
optimal_log_loss_list=[]
for i,d in enumerate(d_list):
    optimal_log_loss=[]
    print(i,d)
    w_opt=w_opt_list[i]
    eps_opt=eps_opt_list[i]
    mu=param_list[i][0]
    sigma = np.diag(param_list[i][1])
    sigma_inv = np.linalg.inv(sigma)
    # Gaussian params
    for j in range(len(w_opt)):
        w=w_opt[j]
        z=eps_opt[j]*(w_opt[j]/(np.linalg.norm(w_opt[j])))
        y_bar_pos=2.0*np.dot(np.dot(mu-z,sigma_inv),mu-z)
        scale_factor_pos=4.0*np.dot(np.dot(mu-z,sigma_inv),mu-z)
        def f_pos(x):
            f_1=np.log(1+np.exp(-1.0*np.sqrt(scale_factor_pos)*x-1.0*y_bar_pos))
            f_2=(1/np.sqrt(2.0*np.pi))*np.exp((-1*(x)**2)/(2.0))
            return f_1*f_2
        loss=integrate.quad(f_pos,-10,10)[0]
        optimal_log_loss.append(loss)
    optimal_log_loss_list.append(optimal_log_loss)
    
print(high_eps_loss_list)
print(optimal_log_loss_list)

PIK8 = "high_eps_loss.dat"
with open(PIK8, "wb") as f:
    pickle.dump(high_eps_loss_list, f)
PIK9 = "optimal_log_loss.dat"
with open(PIK9, "wb") as f:
    pickle.dump(optimal_log_loss_list, f)