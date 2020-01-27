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
	if args.dropping:
		if args.dropping_strat == 'matched':
			model_name += '_matching'
		elif args.dropping_strat == 'approx':
			model_name += '_approx' + str(args.drop_thresh)
	model_name_base = model_name
	if args.trial_num is not None:
		model_name += '_tr' + str(args.trial_num)

	return model_name, model_name_base


def init_dirs(args, train=True):
	model_name, model_name_base = model_naming(args)
	model_dir_name = args.checkpoint_path + '/' + args.dataset_in + '/' + model_name
	log_dir_name = 'logs' + '/' + args.dataset_in + '/' + model_name
	figure_dir_name = 'images' + '/' + args.attack + '/' + args.dataset_in + '/' + model_name
	# if args.track_hard:
	training_output_dir_name = 'training_output' + '/' + args.dataset_in + '/' + model_name
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
		matching_file_name = args.matching_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_' + args.norm + '_cost_zero_' + '{0:.1f}.npy'.format(args.epsilon)
	else:
		matching_file_name = args.matching_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(num_samples) + '_' + args.dataset_in + '_test_' + args.norm + '_cost_zero_' + '{0:.1f}.npy'.format(args.epsilon)
	return matching_file_name

def degree_file_name(args, class_1, class_2, train_data, num_samples):
	if train_data:
		degree_file_name = args.degree_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm + '_' + '{0:.1f}.json'.format(args.epsilon)
	else:
		degree_file_name = args.degree_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_test_' + args.norm + '_' + '{0:.1f}.json'.format(args.epsilon)

	return degree_file_name
