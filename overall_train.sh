
python train_robust.py \
	--dataset_in ${1} \
	--n_classes ${2} \
	--num_samples ${3} \
	--model ${4} \
	--conv_expand ${5} \
	--fc_expand ${6} \
	--depth ${7} \
	--width ${8} \
	--num_of_trials ${9} \
	--batch_size 128 \
	--test_batch_size 128 \
	--train_epochs ${10} \
	--learning_rate ${11} \
	--lr_schedule ${12} \
	--curriculum ${13} \
	--loss_fn ${14} \
	--weight_decay 2e-4 \
	--attack ${15} \
	--epsilon ${16} \
	--attack_iter ${17} \
	--gamma ${18} \
	--clip_min 0.0 \
	--clip_max 1.0 \
	--eps_schedule ${19} \
	--num_restarts ${20} \
	--last_epoch 0 \
	--curr_epoch 0 \
	--checkpoint_path trained_models \
	--marking_strat ${21} \
	--matching_path matchings \
	--degree_path graph_data/degree_results \
	--norm ${22} \
	--drop_thresh ${23} \
	--drop_eps ${24} \
	${25} `#is_adv` \
	${26} `#targeted` \
	${27} `#rand_init` \
	${28} `#save_checkpoint` \
	${29} `#load_checkpoint` \
	${30} `#track_hard` \
	${31} `#is_dropping`