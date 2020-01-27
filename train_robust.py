import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

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
from torch.utils.tensorboard import SummaryWriter

from utils.mnist_models import cnn_3l, cnn_3l_bn, lenet5
from utils.cifar10_models import WideResNet
from utils.densenet_model import DenseNet
from utils.train_utils import train_one_epoch, robust_train_one_epoch, update_hyparam, eps_scheduler
from utils.test_utils import test, robust_test_during_train
from utils.data_utils import load_dataset, load_dataset_custom
from utils.io_utils import init_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument('--dataset_in', type=str, default='MNIST')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=4000)

    # Model args
    parser.add_argument('--model', type=str, default='cnn_3l', choices=['wrn','cnn_3l', 'cnn_3l_bn', 'dn', 'lenet5'])
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
    parser.add_argument('--lr_schedule', type=str, default='linear0')
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
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--curr_epoch', type=int, default=0)
    parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true')
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true')
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
    if args.num_samples is None:
        args.num_samples = 'All'
    model_dir_name, log_dir_name, _, training_output_dir_name = init_dirs(args, train=True)
    if args.save_checkpoint:
        writer = SummaryWriter(log_dir=log_dir_name)
    print('Training %s' % model_dir_name)

    if torch.cuda.is_available():
        print('CUDA enabled')
    else:
        raise ValueError('Needs a working GPU!')
    
    if args.n_classes != 10:
        loader_train, loader_test, data_details = load_dataset_custom(args, data_dir='data')
    else:
        loader_train, loader_test, data_details = load_dataset(args, data_dir='data')

    num_channels = data_details['n_channels']

    if 'MNIST' in args.dataset_in:
        if 'cnn_3l_bn' in args.model:
            net = cnn_3l_bn(args.n_classes, args.conv_expand, args.fc_expand)
        elif 'cnn_3l' in args.model:
            net = cnn_3l(args.n_classes, args.conv_expand, args.fc_expand)
        elif 'lenet5' in args.model:
            net = lenet5(args.n_classes, args.conv_expand, args.fc_expand)
        # elif 'cnn_4l' in args.model:
        #     net = cnn_4l(args.n_classes, args.conv_expand, args.fc_expand)
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

    if 'MNIST' in args.dataset_in:
        summary(net, (1,28,28))
    
    if args.load_checkpoint:
        print('Loading from Epoch %s' % args.curr_epoch)
        ckpt_path = 'checkpoint_' + str(args.last_epoch)
        net.load_state_dict(torch.load(model_dir_name + ckpt_path))
        robust_test_during_train(net, loader_test, args, n_batches=10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    for epoch in range(args.curr_epoch, args.train_epochs):
        start_time = time.time()
        lr = update_hyparam(epoch, args)
        optimizer.param_groups[0]['lr'] = lr
        print('Current learning rate: {}'.format(lr))
        eps, delta = eps_scheduler(epoch, args)
        if not args.is_adv:
            curr_loss = train_one_epoch(net, criterion, optimizer, 
                                  loader_train, args, verbose=False)
            ben_loss = curr_loss
        else:
            curr_loss, ben_loss = robust_train_one_epoch(net, criterion, 
                                    optimizer, loader_train, args, eps, delta, 
                                    epoch, training_output_dir_name, verbose=False)
        print('time_taken for #{} epoch = {:.3f}'.format(epoch+1, time.time()-start_time))
        print('Test set validation')
        acc, acc_adv, test_loss, test_loss_adv = robust_test_during_train(net, 
            criterion, loader_test, args, n_batches=10)
        print('Training set validation')
        acc, acc_adv, test_loss, test_loss_adv = robust_test_during_train(net, 
            criterion, loader_train, args)
        ckpt_path = 'checkpoint_' + str(args.last_epoch)
        if args.save_checkpoint:
            torch.save(net.state_dict(), model_dir_name + ckpt_path)
            writer.add_scalar('Loss/train', curr_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Loss/test_adv', test_loss_adv, epoch)
            writer.add_scalar('Acc/test', acc, epoch)
            writer.add_scalar('Acc/test_adv', acc_adv, epoch)
            writer.add_scalar('Lr', lr, epoch)
            if args.is_adv:
                writer.add_scalar('Loss/train_ben', ben_loss, epoch)
            else:
                writer.add_scalar('Loss/train_ben', 0, epoch)
        print('Train loss - Adv: %s Ben: %s; Test loss - Adv: %s; Ben: %s' %
            (curr_loss, ben_loss, test_loss_adv, test_loss))