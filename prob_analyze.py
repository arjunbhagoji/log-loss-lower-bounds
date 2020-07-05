import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

from utils.io_utils import test_argparse, model_naming, test_probs_save_name, logloss_file_save_name, model_naming_no_eps

parser = test_argparse()

args = parser.parse_args()

args.eps_step = args.epsilon*args.gamma/args.attack_iter
args.new_eps_step = args.new_epsilon*args.gamma/args.new_attack_iter
attack_params = {'attack': args.new_attack, 'epsilon': args.new_epsilon, 
             'attack_iter': args.new_attack_iter, 'eps_step': args.new_eps_step,
             'targeted': args.targeted, 'clip_min': args.clip_min,
             'clip_max': args.clip_max,'rand_init': args.new_rand_init, 
             'num_restarts': args.new_num_restarts}

# Loading optimal loss

class_1 = 3
class_2 = 7
save_file_name = 'logloss_' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm

optimal = np.loadtxt('graph_data/optimal_probs/' + save_file_name + '_' + str(args.epsilon) + '.txt')

optimal_loss=np.zeros((2*args.num_samples))
optimal_loss[:args.num_samples]=-1*np.log(optimal[:args.num_samples,0])
optimal_loss[args.num_samples:]=-1*np.log(optimal[args.num_samples:,1])

wrong_count = 0
wrong_count += len(np.where(optimal[:args.num_samples,0]<=0.5)[0])
wrong_count += len(np.where(optimal[args.num_samples:,1]<=0.5)[0])
zero_one_loss = wrong_count/(2*args.num_samples)

optimal_loss_scalar = np.mean(optimal_loss)

print('Optimal 0-1 loss at eps %s is %s' % (args.epsilon, zero_one_loss))
print('Optimal log-loss at eps %s is %s' % (args.epsilon, optimal_loss_scalar))

# Loading empirical loss

empirical_loss_scalars = np.zeros((args.num_of_trials))

for trial_num in range(1,args.num_of_trials+1):
	args.trial_num = trial_num
	_, model_name = model_naming(args)
	probs_output_fname = test_probs_save_name(args,model_name)
	empirical = np.loadtxt(probs_output_fname + '_train_tr{}.txt'.format(trial_num))
	empirical_loss=np.zeros((2*args.num_samples))
	empirical_loss[:args.num_samples] = -1*np.log(empirical[:args.num_samples,0])
	empirical_loss[args.num_samples:] = -1*np.log(empirical[args.num_samples:,1])
	empirical_loss_scalars[trial_num-1] = np.mean(empirical_loss)

	if trial_num==1:
		# Plotting empirical vs optimal log-loss
		x=np.linspace(0,4,2000)
		# Optimal incorrect
		y=np.tile(np.log(2),2000)
		plt.scatter(optimal_loss, empirical_loss)
		plt.title('Opt. vs emp. logloss at eps: %s' % args.epsilon)
		plt.xlabel('optimal')
		plt.ylabel('empirical')
		plt.plot(x,x,color='red')
		plt.plot(y,x,color='green')
		image_file_name = 'images/opt_vs_emp/' + args.dataset_in + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.norm + '_' + str(args.epsilon) + '.pdf'
		plt.savefig(image_file_name)

	if args.save_test:
		test_output_fname = logloss_file_save_name(args,model_name)
		_, no_eps_model_name = model_naming_no_eps(args)
		overall_test_output_fname = logloss_file_save_name(args,no_eps_model_name)
		f = open(test_output_fname, mode='a')
		if trial_num == 1:
			f.write('tr, adv_tr_loss \n')
		f.write('{}, {} \n'.format(trial_num,np.mean(empirical_loss)))
		if trial_num == args.num_of_trials:
			f.write('{}, {} \n'.format(0,np.mean(empirical_loss_scalars))) 
			f2 = open(overall_test_output_fname, mode='a')
			f2.write('{}, {} \n'.format(args.epsilon,np.mean(empirical_loss_scalars)))
			f2.close()
		f.close()


# print(empirical_loss_scalars)