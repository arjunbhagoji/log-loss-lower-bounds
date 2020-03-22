import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import time
import argparse
from torchsummary import summary

from utils.mnist_models import cnn_3l, cnn_3l_bn
from utils.cifar10_models import WideResNet
from utils.densenet_model import DenseNet
from utils.test_utils import test, robust_test, robust_test_hybrid
from utils.data_utils import load_dataset, load_dataset_custom
from utils.io_utils import init_dirs, model_naming


if __name__ == '__main__':
    torch.random.manual_seed(7)

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
    parser.add_argument('--trial_num', type=int, default=None)
    # parser.add_argument('--learning_rate', type=float, default=0.1)
    # parser.add_argument('--weight_decay', type=float, default=2e-4)

    # Defense args
    parser.add_argument('--is_adv', dest='is_adv', action='store_true')
    parser.add_argument('--attack', type=str, default='PGD_l2',
                        choices=['PGD_l2', 'PGD_linf', 'PGD_l2_hybrid_seed', 'PGD_l2_hybrid_replace'])
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--eps_step', type=float, default=2.0)
    parser.add_argument('--is_dropping', dest='dropping', action='store_true')
    parser.add_argument('--rand_init', dest='rand_init', action='store_true')
    parser.add_argument('--eps_schedule', type=int, default=0)
    parser.add_argument('--num_restarts', type=int, default=1)
    parser.add_argument('--dropping_strat', type=str, default='matched')
    parser.add_argument('--matching_path', type=str, default='matchings')
    parser.add_argument('--degree_path', type=str, default='degree_results')
    parser.add_argument("--norm", default='l2', help="norm to be used")
    parser.add_argument('--drop_thresh', type=int, default=100)
    parser.add_argument('--track_hard', dest='track_hard', action='store_true')

    # Attack args
    parser.add_argument('--new_attack', type=str, default='PGD_l2',
                        choices=['PGD_l2', 'PGD_linf', 'PGD_l2_hybrid_seed', 'PGD_l2_hybrid_replace'])
    parser.add_argument('--new_epsilon', type=float, default=2.0)
    parser.add_argument('--new_attack_iter', type=int, default=20)
    parser.add_argument('--new_eps_step', type=float, default=0.25)
    parser.add_argument('--targeted', dest='targeted', action='store_true')
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--new_rand_init',
                        dest='new_rand_init', action='store_true')
    parser.add_argument('--new_num_restarts', type=int, default=1)

    # IO args
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str,
                        default='trained_models')
    parser.add_argument('--is_viz', dest='viz', action='store_true')

    if torch.cuda.is_available():
        print('CUDA enabled')
    else:
        raise ValueError('Needs a working GPU!')

    args = parser.parse_args()
    model_dir_name, log_dir_name, figure_dir_name, _ = init_dirs(
        args, train=False)
    _, model_name = model_naming(args)
    print('Loading %s' % model_dir_name)

    # Setting to False to load all training data
    args.dropping = False

    if args.n_classes != 10:
        loader_train, loader_test, data_details = load_dataset_custom(
            args, data_dir='data')
    else:
        loader_train, loader_test, data_details = load_dataset(
            args, data_dir='data')

    num_channels = data_details['n_channels']

    attack_params = {'attack': args.new_attack, 'epsilon': args.new_epsilon, 
                     'attack_iter': args.new_attack_iter, 'eps_step': args.new_eps_step,
                     'targeted': args.targeted, 'clip_min': args.clip_min,
                     'clip_max': args.clip_max,'rand_init': args.new_rand_init, 
                     'num_restarts': args.new_num_restarts}

    if 'MNIST' in args.dataset_in:
        if 'cnn_3l_bn' in args.model:
            net = cnn_3l_bn(args.n_classes, args.conv_expand, args.fc_expand)
        elif 'cnn_3l' in args.model:
            net = cnn_3l(args.n_classes, args.conv_expand, args.fc_expand)
        elif 'wrn' in args.model:
            net = WideResNet(depth=args.depth, num_classes=args.n_classes,
                             widen_factor=args.width, input_channels=num_channels)
        elif 'dn' in args.model:
            net = DenseNet(growthRate=12, depth=35, reduction=1.0,
                           bottleneck=False, nClasses=args.n_classes, ChannelsIn=num_channels)
    elif args.dataset_in == 'CIFAR-10':
        if 'wrn' in args.model:
            net = WideResNet(depth=args.depth, num_classes=args.n_classes,
                             widen_factor=args.width, input_channels=num_channels)
        elif 'dn' in args.model:
            net = DenseNet(growthRate=12, depth=100, reduction=0.5,
                           bottleneck=True, nClasses=args.n_classes, ChannelsIn=num_channels)

    if 'linf' in args.attack:
        args.epsilon /= 255.
        args.eps_step /= 255.

    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        net = nn.DataParallel(net)

    args.batch_size = args.batch_size * torch.cuda.device_count()
    print("Using batch size of {}".format(args.batch_size))

    net.cuda()

   # if args.dataset_in == 'MNIST':
    # summary(net, (1,28,28))

    criterion = nn.CrossEntropyLoss(reduction='none')

    net.eval()
    ckpt_path = 'checkpoint_' + str(args.last_epoch)
    net.load_state_dict(torch.load(model_dir_name + ckpt_path))
    print('Using attack {}'.format(args.new_attack))
    if 'hybrid' in args.attack:
        f_eval = robust_test_hybrid
    else:
        f_eval = robust_test
    # Ensuring hard points are tracked for hybrid attack
    n_batches_eval = 0
    args.track_hard = True
    print('Training data results')
    # test(net, loader_train, figure_dir_name)
    acc_train, acc_train_adv, _, _ = f_eval(net, 
        criterion, loader_train, args, attack_params, 0, None, 
        figure_dir_name, n_batches=n_batches_eval, train_data=True, training_time=False)
    print('Test data results')
    # test(net, loader_test, figure_dir_name)
    acc_test, acc_test_adv, _, _ = f_eval(net, 
        criterion, loader_test, args, attack_params, 0, None, 
        figure_dir_name, n_batches=n_batches_eval, train_data=False, training_time=False)

    # Saving test output
    save_test = True
    if save_test:
        test_output_dir = 'test_output' + '/' + args.dataset_in
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        test_output_fname = test_output_dir + '/' + model_name + '_' + args.new_attack
        if args.eps_step != args.new_eps_step or args.attack_iter != args.new_attack_iter:
            test_output_fname += '_delta' + \
                str(args.new_eps_step) + '_t' + str(args.new_attack_iter)
        if args.new_num_restarts != 1:
            test_output_fname += '_restart' + str(args.new_num_restarts)
        test_output_fname += '.txt'
        f = open(test_output_fname, mode='a')
        if args.trial_num == 1:
            f.write('tr, ben_tr, adv_tr, ben_te, adv_te \n')
        f.write('{}, {}, {}, {}, {} \n'.format(args.trial_num,
                                               acc_train, acc_train_adv, acc_test, acc_test_adv))
        f.close()
