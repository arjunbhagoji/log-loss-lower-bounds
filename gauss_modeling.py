



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

X_c1 = X[:num_samples].reshape(num_samples, DATA_DIM)
X_c2 = X[num_samples:].reshape(num_samples, DATA_DIM)