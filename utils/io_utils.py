import os

def init_dirs(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	if 'cnn_3l' in args.model:
		if args.conv_expand != 1 or args.fc_expand != 1:
			model_name += '-conv' + str(args.conv_expand) + '-fc' + str(args.fc_expand) 
	if args.is_adv:
		model_name += '_robust' + '_eps' + str(args.epsilon) + '_k' + str(args.attack_iter) + '_delta' + str(args.eps_step)
	if args.eps_schedule != 0:
		model_name += '_eps_sched' + str(args.eps_schedule)
	# if args.is_adv:
	# 	model_name += '_robust' + '_eps' + str(args.epsilon)
	if args.rand_init:
		model_name += '_rand'
	if args.n_classes != 10:
		model_name += '_cl' + str(args.n_classes)
	if args.dropping:
		model_name += '_matching' 
	model_dir_name = args.checkpoint_path + '_' + args.dataset_in + '/' + model_name
	log_dir_name = 'logs_' + args.dataset_in + '/' + model_name
	figure_dir_name = 'images_' + args.dataset_in + '/' + model_name
	if not os.path.exists(model_dir_name):
		os.makedirs(model_dir_name)
	if not os.path.exists(log_dir_name):
		os.makedirs(log_dir_name)
	if not os.path.exists(figure_dir_name):
		os.makedirs(figure_dir_name)
	model_dir_name += '/'
	log_dir_name += '/'
	figure_dir_name += '/'
	return model_dir_name, log_dir_name, figure_dir_name
		
def matching_file_name(args, class_1, class_2):
    matching_file_name = args.matching_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm + '_cost_zero_' + '{0:.1f}.npy'.format(args.epsilon)

    return matching_file_name