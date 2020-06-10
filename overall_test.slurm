#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --job-name="mnist_test"
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=abhagoji@princeton.edu

# print some info for context
pwd
hostname
date

echo starting job...

# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
source ~/.bash_profile
#conda init
conda activate pytor3

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

python test_robust.py \
	--dataset_in ${1} \
	--n_classes ${2} \
	--num_samples ${3} \
	--model ${4} \
	--conv_expand ${5} \
	--fc_expand ${6} \
	--depth ${7} \
	--width ${8} \
	--batch_size 128 \
	--test_batch_size 128 \
	--lr_schedule ${9} \
	--attack ${10} \
	--epsilon ${11} \
	--attack_iter ${12} \
	--gamma ${13} \
	--clip_min 0.0 \
	--clip_max 1.0 \
	--eps_schedule ${14} \
	--num_restarts ${15} \
	--last_epoch 0 \
	--checkpoint_path trained_models \
	--marking_strat ${16} \
	--matching_path matchings \
	--degree_path graph_data/degree_results \
	--norm ${17} \
	--drop_thresh ${18} \
	--new_attack ${19} \
	--new_epsilon ${20} \
	--new_attack_iter ${21} \
	--new_gamma ${22} \
	--new_num_restarts ${23} \
	--new_marking_strat ${24} \
	--num_of_trials ${25} \
	${26} `#is_adv` \
	${27} `#targeted` \
	${28} `#rand_init` \
	${29} `#is_dropping` \
	${30} `#is_viz` \
	${31} `#save_test` \
	${32} `#track_hard` \
	${33} `#new_rand_init` \