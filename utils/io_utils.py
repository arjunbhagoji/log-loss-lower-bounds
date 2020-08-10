import os
import argparse

def test_argparse():
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument('--dataset_in', type=str, default='MNIST')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=None)

    # Model args
    parser.add_argument('--model', type=str, default='cnn_3l',
                        choices=['wrn', 'cnn_3l', 'cnn_3l_bn', 'dn'])
    parser.add_argument('--conv_expand', type=int, default=1)
    parser.add_argument('--fc_expand', type=int, default=1)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--lr_schedule', type=str, default='linear0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    # parser.add_argument('--learning_rate', type=float, default=0.1)
    # parser.add_argument('--weight_decay', type=float, default=2e-4)

    # Defense args
    parser.add_argument('--is_adv', dest='is_adv', action='store_true')
    parser.add_argument('--attack', type=str, default='PGD_l2',
                        choices=['PGD_l2', 'PGD_linf', 'PGD_l2_hybrid_seed', 'PGD_l2_hybrid_replace'])
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--eps_step', type=float, default=2.0)
    parser.add_argument('--is_dropping', dest='dropping', action='store_true')
    parser.add_argument('--rand_init', dest='rand_init', action='store_true')
    parser.add_argument('--eps_schedule', type=int, default=0)
    parser.add_argument('--num_restarts', type=int, default=1)
    parser.add_argument('--marking_strat', type=str, default=None)
    parser.add_argument('--matching_path', type=str, default='matchings')
    parser.add_argument('--degree_path', type=str, default='graph_data/degree_results')
    parser.add_argument("--norm", default='l2', help="norm to be used")
    parser.add_argument('--drop_thresh', type=int, default=100)
    parser.add_argument('--curriculum', type=str, default='all')

    # Attack args
    parser.add_argument('--new_attack', type=str, default='PGD_l2',
                        choices=['PGD_l2', 'PGD_linf', 'PGD_l2_hybrid_seed', 'PGD_l2_hybrid_replace'])
    parser.add_argument('--new_epsilon', type=float, default=2.0)
    parser.add_argument('--new_attack_iter', type=int, default=20)
    parser.add_argument('--new_gamma', type=float, default=1.0)
    parser.add_argument('--targeted', dest='targeted', action='store_true')
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--new_rand_init',
                        dest='new_rand_init', action='store_true')
    parser.add_argument('--new_num_restarts', type=int, default=1)
    parser.add_argument('--new_marking_strat', type=str, default=None)

    # IO args
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str,
                        default='trained_models')
    parser.add_argument('--is_viz', dest='viz', action='store_true')
    
    # Trial args
    parser.add_argument('--num_of_trials', type=int, default=1)
    parser.add_argument('--save_test', dest='save_test', action='store_true')
    parser.add_argument('--track_hard', dest='track_hard', action='store_true')

    return parser


def model_naming(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	elif 'resnet' in args.model:
		model_name += str(args.depth)
	elif 'cnn_3l' in args.model or 'lenet5' in args.model:
		if args.conv_expand != 1 or args.fc_expand != 1:
			model_name += '-conv' + str(args.conv_expand) + '-fc' + str(args.fc_expand) 
	if args.loss_fn != 'CE':
		model_name += '_' + args.loss_fn
	if args.lr_schedule != 'linear0':
		model_name += '_lr-sch' + str(args.lr_schedule)
	if args.learning_rate != 0.1:
		model_name += '_lr' + str(args.learning_rate)
	if args.attack != 'PGD_l2' and args.attack != 'PGD_linf':
		model_name += '_' + str(args.attack)
		if args.marking_strat != 'matched':
			model_name += '_' + str(args.marking_strat)
	if args.is_adv:
		model_name += '_robust' + '_eps' + str(args.epsilon) + '_k' + str(args.attack_iter) + '_delta' + str(args.eps_step)
	if args.eps_schedule != 0:
		model_name += '_eps_sched' + str(args.eps_schedule)
	if args.rand_init:
		model_name += '_rand'
	if args.n_classes != 10:
		model_name += '_cl' + str(args.n_classes)
	if args.num_samples != 2000:
		model_name += '_ns' + str(args.num_samples)
	if args.num_restarts != 1:
		model_name += '_restart' + str(args.num_restarts)
	if args.dropping:
		if args.marking_strat == 'matched':
			model_name += '_matching'
		elif args.marking_strat == 'approx':
			model_name += '_approx' + str(args.drop_thresh)
		elif args.marking_strat == 'random':
			model_name += '_random' + str(args.drop_thresh)
	if args.curriculum != 'all':
		model_name += '_curr' + str(args.curriculum)
	model_name_base = model_name
	if args.trial_num is not None:
		model_name += '_tr' + str(args.trial_num)

	return model_name, model_name_base

def model_naming_no_eps(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	if 'cnn_3l' in args.model or 'lenet5' in args.model:
		if args.conv_expand != 1 or args.fc_expand != 1:
			model_name += '-conv' + str(args.conv_expand) + '-fc' + str(args.fc_expand) 
	if args.lr_schedule != 'linear0':
		model_name += '_lr-sch' + str(args.lr_schedule)
	if args.learning_rate != 0.1:
		model_name += '_lr' + str(args.learning_rate)
	if args.attack != 'PGD_l2' and args.attack != 'PGD_linf':
		model_name += '_' + str(args.attack)
		if args.marking_strat != 'matched':
			model_name += '_' + str(args.marking_strat)
	# if args.is_adv:
	# 	model_name += '_robust' + '_eps' + str(args.epsilon) + '_k' + str(args.attack_iter) + '_delta' + str(args.eps_step)
	if args.eps_schedule != 0:
		model_name += '_eps_sched' + str(args.eps_schedule)
	if args.rand_init:
		model_name += '_rand'
	if args.n_classes != 10:
		model_name += '_cl' + str(args.n_classes)
	if args.num_samples != 2000:
		model_name += '_ns' + str(args.num_samples)
	if args.num_restarts != 1:
		model_name += '_restart' + str(args.num_restarts)
	if args.dropping:
		if args.marking_strat == 'matched':
			model_name += '_matching'
		elif args.marking_strat == 'approx':
			model_name += '_approx' + str(args.drop_thresh)
		elif args.marking_strat == 'random':
			model_name += '_random' + str(args.drop_thresh)
	if args.curriculum != 'all':
		model_name += '_curr' + str(args.curriculum)
	model_name_base = model_name
	if args.trial_num is not None:
		model_name += '_tr' + str(args.trial_num)

	return model_name, model_name_base



def init_dirs(args, train=True):
	model_name, model_name_base = model_naming(args)
	model_dir_name = args.checkpoint_path + '/' + args.dataset_in + '/' + model_name_base + '/' + model_name
	log_dir_name = 'logs' + '/' + args.dataset_in + '/' + model_name_base + '/' + model_name
	figure_dir_name = 'images' + '/' + args.attack + '/' + args.dataset_in + '/' + model_name_base + '/' + model_name
	# if args.track_hard:
	training_output_dir_name = 'training_output' + '/' + args.dataset_in + '/' + model_name_base + '/' + model_name
	if train:
		if args.save_checkpoint:
			if not os.path.exists(model_dir_name):
				os.makedirs(model_dir_name)
			if not os.path.exists(log_dir_name):
				os.makedirs(log_dir_name)
		if args.track_hard:
			if not os.path.exists(training_output_dir_name):
				os.makedirs(training_output_dir_name)
	training_output_dir_name += '/'
	model_dir_name += '/'
	log_dir_name += '/'
	return model_dir_name, log_dir_name, figure_dir_name, training_output_dir_name
		
def matching_file_name(args, class_1, class_2, train_data, num_samples):
	if train_data:
		matching_file_name = args.matching_path + '/' + args.dataset_in + '/' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_' + args.norm + '_cost_zero_' + '{0:.1f}.npy'.format(args.epsilon)
	else:
		matching_file_name = args.matching_path + '/' + args.dataset_in + '/' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_test_' + args.norm + '_cost_zero_' + '{0:.1f}.npy'.format(args.epsilon)
	return matching_file_name

def global_matching_file_name(args, class_1, class_2, train_data, num_samples):
	if train_data:
		global_matching_file_name = args.matching_path + '/' + args.dataset_in + '/' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm
	else:
		global_matching_file_name = args.matching_path + '/' + args.dataset_in + '/' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_test_' + args.norm
	
	global_dict_name = global_matching_file_name + '_global_dict.json'
	global_tuple_name = global_matching_file_name + '_global.npy'

	return global_dict_name, global_tuple_name

def degree_file_name(args, class_1, class_2, train_data, num_samples):
	if train_data:
		degree_file_name = args.degree_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm + '_' + '{0:.1f}.json'.format(args.epsilon)
	else:
		degree_file_name = args.degree_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_test_' + args.norm + '_' + '{0:.1f}.json'.format(args.epsilon)

	return degree_file_name

def distance_file_name(args, class_1, class_2, train_data, num_samples):
	if train_data:
		dist_file_name = 'distances/' + args.dataset_in + '_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm + '.npy'
	else:
		dist_file_name = 'distances/' + args.dataset_in + '_test_' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.norm + '.npy'

	return dist_file_name

def test_file_save_name(args, model_name):
	test_output_dir = 'test_output' + '/' + args.dataset_in
	if not os.path.exists(test_output_dir):
	    os.makedirs(test_output_dir)
	test_output_fname = test_output_dir + '/' + model_name + '_' + args.new_attack
	if 'hybrid' in args.new_attack:
	    test_output_fname += '_mark_' + args.new_marking_strat
	if args.eps_step != args.new_eps_step or args.attack_iter != args.new_attack_iter:
	    test_output_fname += '_delta' + \
	        str(args.new_eps_step) + '_t' + str(args.new_attack_iter)
	if args.new_num_restarts != 1:
	    test_output_fname += '_restart' + str(args.new_num_restarts)
	test_output_fname += '.txt'

	return test_output_fname

def logloss_file_save_name(args, model_name):
	test_output_dir = 'test_output' + '/' + args.dataset_in + '_logloss'
	if not os.path.exists(test_output_dir):
	    os.makedirs(test_output_dir)
	test_output_fname = test_output_dir + '/' + model_name + '_' + args.new_attack
	if 'hybrid' in args.new_attack:
	    test_output_fname += '_mark_' + args.new_marking_strat
	if args.eps_step != args.new_eps_step or args.attack_iter != args.new_attack_iter:
	    test_output_fname += '_delta' + \
	        str(args.new_eps_step) + '_t' + str(args.new_attack_iter)
	if args.new_num_restarts != 1:
	    test_output_fname += '_restart' + str(args.new_num_restarts)
	test_output_fname += '.txt'

	return test_output_fname

def test_probs_save_name(args, model_name):
	test_output_dir = 'probs_output' + '/' + args.dataset_in
	if not os.path.exists(test_output_dir):
	    os.makedirs(test_output_dir)
	test_output_fname = test_output_dir + '/' + model_name + '_' + args.new_attack
	if 'hybrid' in args.new_attack:
	    test_output_fname += '_mark_' + args.new_marking_strat
	if args.eps_step != args.new_eps_step or args.attack_iter != args.new_attack_iter:
	    test_output_fname += '_delta' + \
	        str(args.new_eps_step) + '_t' + str(args.new_attack_iter)
	if args.new_num_restarts != 1:
	    test_output_fname += '_restart' + str(args.new_num_restarts)
	# test_output_fname += '.txt'

	return test_output_fname