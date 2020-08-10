import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np
import time
import collections
import json 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack, pgd_l2_attack, hybrid_attack
from .data_utils import load_dataset_tensor
from .loss_utils import KL_loss_flat, trades_loss

class_1 = 3
class_2 = 7

def eps_scheduler(epoch, args):
    eps_scale = (args.eps_step*args.attack_iter)/args.epsilon
    if args.eps_schedule == 1:
        print(epoch, args.train_epochs/2)
        if epoch <= args.train_epochs/2:
          eps = args.epsilon/2
          delta = eps*eps_scale/args.attack_iter
        elif epoch > args.train_epochs/2:
          eps = args.epsilon
          delta = args.eps_step
    elif args.eps_schedule == 0:
        eps = args.epsilon
        delta = args.eps_step
    return eps, delta

def update_hyparam(epoch, args):
    #args.learning_rate = args.learning_rate * (0.6 ** ((max((epoch-args.schedule_length), 0) // 5)))
    if 'linear' in args.lr_schedule:
      if '0' in args.lr_schedule:
        lr_steps = [100, 150, 200]
      elif '1' in args.lr_schedule:
        lr_steps = [50, 100, 150, 200]
      lr = args.learning_rate
      for i in lr_steps:
          if epoch<i:
              break
          lr /= 10
    return lr 


########################################  Natural training ########################################
def train_one_epoch(model, optimizer, loader_train, args, verbose=True):
    losses = []
    model.train()
    for t, (x, y, idx, ez, m) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        scores = model(x_var)
        # loss = loss_fn(scores, y_var)
        if args.loss_fn=='CE':
          loss_function = nn.CrossEntropyLoss(reduction='none') 
          batch_loss = loss_function(scores, y_var)
        loss = torch.mean(batch_loss)
        losses.append(loss.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        # print(model.conv1.weight.grad)
        optimizer.step()
    if verbose:
        print('loss = %.8f' % (loss.data))
    return np.mean(losses)      


########################################  Adversarial training ########################################
def robust_train_one_epoch(model, optimizer, loader_train, args, eps, 
                            delta, epoch, training_output_dir_name, verbose=True):
    print('Current eps: {}, delta: {}'.format(eps, delta))
    losses = []
    losses_ben = []
    model.train()
    if 'hybrid' in args.attack:
      training_time=True
      trainset, testset, data_details = load_dataset_tensor(args, 
                                    data_dir='data', training_time=training_time)
      print('Data loaded for hybrid attack of len {}'.format(len(trainset)))
    if 'KL' in args.loss_fn:
      opt_prob_dir = 'graph_data/optimal_probs/'
      opt_fname = 'logloss_' + str(class_1) + '_' + str(class_2) + '_' + str(args.num_samples) + '_' + args.dataset_in + '_' + args.norm + '_' + str(args.epsilon) + '.txt'
      optimal_scores_overall = np.loadtxt(opt_prob_dir+opt_fname)
    for t, (x, y, idx, ez, m) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        if args.loss_fn == 'trades':
          loss, loss_ben, loss_adv = trades_loss(model, x, y, optimizer, delta, eps, 
                                      args.attack_iter, args.gamma, beta=1.0, distance=args.attack)
          losses.append(loss_adv.data.cpu().numpy())
          losses_ben.append(loss_ben.data.cpu().numpy())
        else:  
          x_mod = None
          if 'hybrid' in args.attack:
            # Find matched and unmatched data and labels
            unmatched_x = x[ez]
            unmatched_y = y[ez]
            matched_x = x[~ez]
            matched_y = y[~ez]
            if 'seed' in args.attack:
              if len(m[~ez]>0):
                # print('Performing hybrid attack')
                x_mod = hybrid_attack(matched_x, ez , m, trainset, eps)
            elif 'replace' in args.attack:
              # Only construct adv. examples for unmatched
              x = x[ez]
              y = y[ez]
          x_var = Variable(x, requires_grad= True)
          y_var = Variable(y, requires_grad= False)
          if args.targeted:
              y_target = generate_target_label_tensor(
                                 y_var.cpu(), args).cuda()
          else:
              y_target = y_var
          if 'PGD_linf' in args.attack:
              adv_x = pgd_attack(model, x, x_var, y_target, args.attack_iter,
                             eps, delta, args.clip_min, args.clip_max, 
                             args.targeted, args.rand_init)
          elif 'PGD_l2' in args.attack:
              adv_x = pgd_l2_attack(model, x, x_var, y_target, args.attack_iter,
                             eps, delta, args.clip_min, args.clip_max, 
                             args.targeted, args.rand_init,
                             args.num_restarts, x_mod, ez)
          if 'hybrid' in args.attack:
            x = torch.cat((unmatched_x,matched_x))
            y = torch.cat((unmatched_y,matched_y))
            y_var = Variable(y, requires_grad= False)
            if 'replace' in args.attack:
              x_mod = hybrid_attack(matched_x, ez, m, rel_data, args.new_epsilon)
              adv_x = torch.cat((adv_x, x_mod))
          scores = model(adv_x)
          if args.loss_fn == 'CE':
            loss_function = nn.CrossEntropyLoss(reduction='none')
            batch_loss_adv = loss_function(scores, y_var)
            batch_loss_ben = loss_function(model(x),y_var)
          elif args.loss_fn == 'KL':
            optimal_scores = torch.from_numpy(optimal_scores_overall[idx]).float().cuda()
            loss_function = nn.KLDivLoss(reduction='none')
            batch_loss_adv = loss_function(scores, optimal_scores)
            batch_loss_ben = loss_function(model(x), optimal_scores)
          elif args.loss_fn == 'KL_flat':
            optimal_scores = torch.from_numpy(optimal_scores_overall[idx]).float().cuda()
            batch_loss_adv = KL_loss_flat(scores, optimal_scores, y_var, t)
            batch_loss_ben = KL_loss_flat(model(x), optimal_scores, y_var, t)
          loss = torch.mean(batch_loss_adv)
          loss_ben = torch.mean(batch_loss_ben)
          losses_ben.append(loss_ben.data.cpu().numpy())
          losses.append(loss.data.cpu().numpy())
          losses_ben.append(loss_ben.data.cpu().numpy())
        # GD step
        optimizer.zero_grad()
        loss.backward()
        # print(model.conv1.weight.grad)
        optimizer.step()
        if verbose:
            print('loss = %.8f' % (loss.data))
    return np.mean(losses), np.mean(losses_ben)
