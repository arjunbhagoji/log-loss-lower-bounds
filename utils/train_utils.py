import torch
from torch.autograd import Variable

import numpy as np
import time 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack, pgd_l2_attack

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
    if args.lr_schedule == 0:
      lr_steps = [100, 150, 200]
    elif args.lr_schedule == 1:
      lr_steps = [50, 100, 150, 200]
    lr = args.learning_rate
    for i in lr_steps:
        if epoch<i:
            break
        lr /= 10
    return lr 


########################################  Natural training ########################################
def train_one_epoch(model, loss_fn, optimizer, loader_train, verbose=True):
    losses = []
    model.train()
    for t, (x, y, z) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        scores = model(x_var)
        # loss = loss_fn(scores, y_var)
        batch_loss = loss_fn(scores, y_var)
        loss = torch.mean(batch_loss)
        losses.append(loss.data.cpu().numpy())
        if args.track_hard:
          easy_idx = np.where(z.data.cpu().numpy()==True)
          hard_idx = np.where(z.data.cpu().numpy()==False)
          batch_loss_hard = batch_loss[hard_idx]
          batch_loss_easy = batch_loss[easy_idx]
          print(batch_loss_hard)
        optimizer.zero_grad()
        loss.backward()
        # print(model.conv1.weight.grad)
        optimizer.step()
    if verbose:
        print('loss = %.8f' % (loss.data))
    return np.mean(losses)      


########################################  Adversarial training ########################################
def robust_train_one_epoch(model, loss_fn, optimizer, loader_train, args, eps, delta, verbose=True):
    print('Current eps: {}, delta: {}'.format(eps, delta))
    losses = []
    losses_ben = []
    if args.track_hard:
      losses_easy = []
      losses_hard = []
      losses_ben_easy = []
      losses_ben_hard = []
    model.train()
    for t, (x, y, z) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        if args.targeted:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
        if 'PGD_linf' in args.attack:
            adv_x = pgd_attack(model,
                           x,
                           x_var,
                           y_target,
                           args.attack_iter,
                           eps,
                           delta,
                           args.clip_min,
                           args.clip_max, 
                           args.targeted,
                           args.rand_init)
        elif 'PGD_l2' in args.attack:
            adv_x = pgd_l2_attack(model,
                           x,
                           x_var,
                           y_target,
                           args.attack_iter,
                           eps,
                           delta,
                           args.clip_min,
                           args.clip_max, 
                           args.targeted,
                           args.rand_init)
        
        scores = model(adv_x)
        batch_loss = loss_fn(scores, y_var)
        loss = torch.mean(batch_loss)
        losses.append(loss.data.cpu().numpy())
        batch_loss_ben = loss_fn(model(x),y)
        loss_ben = torch.mean(batch_loss_ben)
        losses_ben.append(loss_ben.data.cpu().numpy())
        if args.track_hard:
          easy_idx = np.where(z.data.cpu().numpy()==True)
          hard_idx = np.where(z.data.cpu().numpy()==False)
          if len(hard_idx[0])>0:
            batch_loss_hard = batch_loss[hard_idx]
            loss_hard = torch.mean(batch_loss_hard)
            losses_hard.append(loss_hard.data.cpu().numpy())
            batch_loss_easy = batch_loss[easy_idx]
            loss_easy = torch.mean(batch_loss_easy)
            losses_easy.append(loss_easy.data.cpu().numpy())
            batch_loss_ben_hard = batch_loss_ben[hard_idx]
            loss_ben_hard = torch.mean(batch_loss_ben_hard)
            losses_ben_hard.append(loss_ben_hard.data.cpu().numpy())
            batch_loss_ben_easy = batch_loss_ben[easy_idx]
            loss_ben_easy = torch.mean(batch_loss_ben_easy)
            losses_ben_easy.append(loss_ben_easy.data.cpu().numpy())
          else:
            losses_hard.append(0.0)
            batch_loss_easy = batch_loss[easy_idx]
            loss_easy = torch.mean(batch_loss_easy)
            losses_easy.append(loss_easy.data.cpu().numpy())
            losses_ben_hard.append(0.0)
            batch_loss_ben_easy = batch_loss_ben[easy_idx]
            loss_ben_easy = torch.mean(batch_loss_ben_easy)
            losses_ben_easy.append(loss_ben_easy.data.cpu().numpy())
        # GD step
        optimizer.zero_grad()
        loss.backward()
        # print(model.conv1.weight.grad)
        optimizer.step()
        if verbose:
            print('loss = %.8f' % (loss.data))
    if args.track_hard:
      print('Adv loss easy: %.8f' % np.mean(losses_easy))
      print('Adv loss hard: %.8f' % np.mean(losses_hard))
      print('Ben loss easy: %.8f' % np.mean(losses_ben_easy))
      print('Ben loss hard: %.8f' % np.mean(losses_ben_hard))
    return np.mean(losses), np.mean(losses_ben)
