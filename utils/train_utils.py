import torch
from torch.autograd import Variable

import numpy as np
import time
import collections
import json 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack, pgd_l2_attack, hybrid_attack
from .data_utils import load_dataset_tensor

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
def train_one_epoch(model, loss_fn, optimizer, loader_train, args, verbose=True):
    losses = []
    model.train()
    for t, (x, y, idx, ez, m) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        scores = model(x_var)
        # loss = loss_fn(scores, y_var)
        batch_loss = loss_fn(scores, y_var)
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
def robust_train_one_epoch(model, loss_fn, optimizer, loader_train, args, eps, 
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
    for t, (x, y, idx, ez, m) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
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
        batch_loss_adv = loss_fn(scores, y_var)
        loss = torch.mean(batch_loss_adv)
        losses.append(loss.data.cpu().numpy())
        batch_loss_ben = loss_fn(model(x),y_var)
        loss_ben = torch.mean(batch_loss_ben)
        losses_ben.append(loss_ben.data.cpu().numpy())
        # GD step
        optimizer.zero_grad()
        loss.backward()
        # print(model.conv1.weight.grad)
        optimizer.step()
        if verbose:
            print('loss = %.8f' % (loss.data))
    return np.mean(losses), np.mean(losses_ben)
