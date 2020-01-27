import numpy as np
import time
import argparse
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from utils.io_utils import init_dirs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Data args
	parser.add_argument('--dataset_in', type=str, default='MNIST')
	parser.add_argument('--n_classes', type=int, default=2)
	parser.add_argument('--num_samples', type=int, default=4000)

	# Model args
	parser.add_argument('--model', type=str, default='cnn_3l', choices=['wrn','cnn_3l', 'dn'])
	parser.add_argument('--conv_expand', type=int, default=1)
	parser.add_argument('--fc_expand', type=int, default=1)
	parser.add_argument('--depth', type=int, default=28)
	parser.add_argument('--width', type=int, default=10)
	parser.add_argument('--trial_num', type=int, default=None)

	# Training args
	parser.add_argument('--batch_size', type=int, default=128) 
	parser.add_argument('--test_batch_size', type=int, default=128)
	parser.add_argument('--train_epochs', type=int, default=20)
	parser.add_argument('--learning_rate', type=float, default=0.1)
	parser.add_argument('--lr_schedule', type=int, default=0)
	parser.add_argument('--weight_decay', type=float, default=2e-4)

	# Attack args
	parser.add_argument('--is_adv', dest='is_adv', action='store_true')
	parser.add_argument('--attack', type=str, default='PGD_l2')
	parser.add_argument('--epsilon', type=float, default=3.0)
	parser.add_argument('--attack_iter', type=int, default=10)
	parser.add_argument('--eps_step', type=float, default=0.3)
	parser.add_argument('--targeted', dest='targeted', action='store_true')
	parser.add_argument('--clip_min', type=float, default=0)
	parser.add_argument('--clip_max', type=float, default=1.0)
	parser.add_argument('--rand_init', dest='rand_init', action='store_true')
	parser.add_argument('--eps_schedule', type=int, default=0)

	# IO args
	# parser.add_argument('--last_epoch', type=int, default=0)
	# parser.add_argument('--curr_epoch', type=int, default=0)
	parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true')
	# parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true')
	parser.add_argument('--checkpoint_path', type=str, default='trained_models')

	# Matching args
	parser.add_argument('--track_hard', dest='track_hard', action='store_true')
	parser.add_argument('--is_dropping', dest='dropping', action='store_true')
	parser.add_argument('--dropping_strat', type=str, default='matched')
	parser.add_argument('--matching_path', type=str, default='matchings')
	parser.add_argument('--degree_path', type=str, default='graph_data/degree_results')
	parser.add_argument("--norm", default='l2', help="norm to be used")
	parser.add_argument('--drop_thresh', type=int, default=100)

	args = parser.parse_args()
	model_dir_name, log_dir_name, _, training_output_dir_name = init_dirs(args)

	f_name = training_output_dir_name + 'losses.json'
	loss_data = []
	for line in open(f_name, 'r'):
		loss_data.append(json.loads(line))
	num_epochs = len(loss_data)

	# fig = plt.figure(figsize=plt.figaspect(.5))
	fig, axarr = plt.subplots(1, 2)
	fig.suptitle('Distribution of loss among easy and hard points', fontsize=16)

	for i in range(num_epochs):
		# ax_loss = fig.add_subplot(1,num_epochs,i+1)

		manual_bins = np.linspace(0,2.5,50)

		hist, bins = np.histogram(loss_data[i]['batch_losses_hard'], bins=manual_bins, density=True)

		x = (bins[:-1] + bins[1:])/2

		width = abs(x[0]-x[1])

		hard_bars = axarr[i].bar(x, hist, width=width, color='red',alpha=0.5)

		hard_avg = axarr[i].axvline(np.mean(loss_data[i]['batch_losses_hard']), color='red', ls='--')

		hist, bins = np.histogram(loss_data[i]['batch_losses_easy'], bins=manual_bins, density=True)

		x = (bins[:-1] + bins[1:])/2

		width = abs(x[0]-x[1])

		easy_bars = axarr[i].bar(x, hist, width=width, color='blue',alpha=0.5)

		easy_avg = axarr[i].axvline(np.mean(loss_data[i]['batch_losses_easy']), color='blue', ls='--')

		if i==0:
			axarr[i].legend((hard_bars[0], easy_bars[0]), ('Hard', 'Easy'))
			axarr[i].set_ylabel('Density')
		
		axarr[i].set_xlabel('Loss')
		axarr[i].set_title('Epoch %s' % i)
	

	# plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig(training_output_dir_name + 'loss_tracker.png',format='png')        

	plt.clf()