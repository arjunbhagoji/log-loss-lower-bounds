import os

def init_dirs(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	# if args.is_adv:
	# 	model_name += '_robust' + '_eps' + str(args.epsilon) + '_k' + str(args.attack_iter) + '_delta' + str(args.eps_step)
	if args.is_adv:
		model_name += '_robust' + '_eps' + str(args.epsilon)
	if args.rand_init:
		model_name += 'rand'
	if args.n_classes != 10:
		model_name += '_cl' + str(args.n_classes)
	if args.dropping:
		model_name += '_matching' 
	model_dir_name = args.checkpoint_path + '_' + args.dataset_in + '/' + model_name
	log_dir_name = 'logs_' + args.dataset_in + '/' +model_name
	if not os.path.exists(model_dir_name):
		os.makedirs(model_dir_name)
	if not os.path.exists(log_dir_name):
		os.makedirs(log_dir_name)
	model_dir_name += '/'
	log_dir_name += '/'
	return model_dir_name, log_dir_name
		
def matching_file_name(args, class_1, class_2):
    matching_file_name = args.matching_path + '/' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm + '_cost_zero_' + '{0:.1f}.npy'.format(args.epsilon)

    return matching_file_name