import os


def model_naming(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	if 'cnn_3l' in args.model or 'lenet5' in args.model:
		if args.conv_expand != 1 or args.fc_expand != 1:
			model_name += '-conv' + str(args.conv_expand) + '-fc' + str(args.fc_expand) 
	if args.lr_schedule != 'linear0':
		model_name += '_lr-sch' + str(args.lr_schedule)
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