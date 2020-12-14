from torch.autograd import Variable
import torch
import torchvision

import numpy as np
import time
import os 
from scipy.special import softmax

import collections
import json 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack, pgd_l2_attack, hybrid_attack
from .image_utils import custom_save_image
from .data_utils import load_dataset_tensor


def hard_point_class_count(pred_dict):
  count_one_wrong = 0
  count_both_wrong = 0
  count_both_correct = 0
  for k,v in pred_dict.items():
    matched_idx = v[0]
    matched_idx_entry = pred_dict[str(matched_idx)]
    assert matched_idx_entry[1] != v[1]
    if v[1]!=v[2] and matched_idx_entry[1]!=matched_idx_entry[2]:
      count_both_wrong += 1
    elif v[1]==v[2] and matched_idx_entry[1]==matched_idx_entry[2]:
      count_both_correct += 1
    else:
      count_one_wrong += 1
  print('Both correct: {}, Both wrong: {}, One wrong: {}'.format(count_both_correct/2, 
          count_both_wrong/2, count_one_wrong/2))

def track_hard_losses(ez_np, batch_loss, batch_loss_ben, loss_dict, t):
  easy_idx = np.where(ez_np==True)
  hard_idx = np.where(ez_np==False)
  if t == 0:
    loss_dict['batch_losses_easy'] = []
    loss_dict['batch_losses_hard'] = []
    loss_dict['batch_losses_ben_easy'] = []
    loss_dict['batch_losses_ben_hard'] = []
  if len(hard_idx[0])>0:
    # Adding adv loss
    batch_loss_hard = batch_loss[hard_idx]
    loss_dict['batch_losses_hard'].extend(batch_loss_hard.data.cpu().numpy().tolist())
    batch_loss_easy = batch_loss[easy_idx]
    loss_dict['batch_losses_easy'].extend(batch_loss_easy.data.cpu().numpy().tolist())
    # Addding benign loss
    batch_loss_ben_hard = batch_loss_ben[hard_idx]
    loss_dict['batch_losses_ben_hard'].extend(batch_loss_ben_hard.data.cpu().numpy().tolist())
    batch_loss_ben_easy = batch_loss_ben[easy_idx]
    loss_dict['batch_losses_ben_easy'].extend(batch_loss_ben_easy.data.cpu().numpy().tolist())
  else:
    batch_loss_easy = batch_loss[easy_idx]
    loss_dict['batch_losses_easy'].extend(batch_loss_easy.data.cpu().numpy().tolist())
    batch_loss_ben_easy = batch_loss_ben[easy_idx]
    loss_dict['batch_losses_ben_easy'].extend(batch_loss_ben_easy.data.cpu().numpy().tolist())

  return loss_dict


def test(model, loader, figure_dir_name):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    steps = 1
    for x, y, z in loader:
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= False)
        y_var = Variable(y, requires_grad= False)        
        scores = model(x_var)
        _, preds = scores.data.max(1)
        num_correct += (preds == y).sum()
        # if args.viz and steps == 1:
        #   torchvision.utils.save_image(x, '{}/clean.jpg'.format(figure_dir_name))

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    return 100.*acc

def robust_test_hybrid(model, loss_fn, loader, args, att_dir, epoch=0, training_output_dir_name=None, 
                      figure_dir_name=None, n_batches=0, train_data=False, 
                      training_time=False):

  """
  n_batches (int): Number of batches for evaluation.
  """
  model.eval()
  num_correct, num_correct_adv, num_samples = 0, 0, 0
  steps = 1

  losses_adv = []
  losses_ben = []
  loss_dict = collections.OrderedDict()
  pred_dict = {}
  prob_dict = {}
  if training_time and args.track_hard:
    f_name = training_output_dir_name + 'losses.json'
    f = open(f_name,'a+')
    loss_dict['epoch'] = epoch
  trainset, testset, data_details = load_dataset_tensor(args, data_dir='data', training_time=training_time)
  if train_data:
    rel_data = trainset
  else:
    rel_data = testset 
  for t, (x, y, idx, ez, m) in enumerate(loader):
      x = x.cuda()
      y = y.cuda()
      x_mod = None
      gen_adv_flag = False
      # Find matched and unmatched data and labels
      unmatched_x = x[ez]
      unmatched_y = y[ez]
      matched_x = x[~ez]
      matched_y = y[~ez]
      if 'seed' in att_dir['attack']:
        gen_adv_flag = True
        # print('Seeding')
        if len(m[~ez])>0:
          x_mod = hybrid_attack(matched_x, ez , m, rel_data, att_dir['epsilon'])
      elif 'replace' in att_dir['attack']:
        if len(unmatched_x)>0:
          gen_adv_flag = True
        # Only construct adv. examples for unmatched
        x = x[ez]
        y = y[ez]
      x_var = Variable(x, requires_grad= True)
      y_var = Variable(y, requires_grad=False)
      if att_dir['targeted']:
          y_target = generate_target_label_tensor(
                             y_var.cpu(), args).cuda()
      else:
          y_target = y_var
      if gen_adv_flag:
        if 'PGD_linf' in att_dir['attack']:
            adv_x = pgd_attack(model, x, x_var, y_target, att_dir['attack_iter'],
                           att_dir['epsilon'], att_dir['eps_step'], att_dir['clip_min'],
                           att_dir['clip_max'], att_dir['targeted'], att_dir['rand_init'])
        elif 'PGD_l2' in att_dir['attack']:
            adv_x = pgd_l2_attack(model, x, x_var, y_target, att_dir['attack_iter'],
                           att_dir['epsilon'], att_dir['eps_step'], att_dir['clip_min'],
                           att_dir['clip_max'], att_dir['targeted'], att_dir['rand_init'], 
                           att_dir['num_restarts'], x_mod, ez)
      x = torch.cat((unmatched_x,matched_x))
      y = torch.cat((unmatched_y,matched_y))
      if 'replace' in att_dir['attack']:
        x_mod = hybrid_attack(matched_x, ez, m, rel_data, att_dir['epsilon'])
        if gen_adv_flag:
          adv_x = torch.cat((adv_x, x_mod))
        else:
          adv_x = x_mod
      # Predictions
      scores = model(x.cuda()) 
      _, preds = scores.data.max(1)
      scores_adv = model(adv_x)
      _, preds_adv = scores_adv.data.max(1)
      # Losses
      batch_loss_adv = loss_fn(scores_adv, y)
      loss_adv = torch.mean(batch_loss_adv)
      losses_adv.append(loss_adv.data.cpu().numpy())
      batch_loss_ben = loss_fn(scores, y)
      loss_ben = torch.mean(batch_loss_ben)
      losses_ben.append(loss_ben.data.cpu().numpy())
      # Correct Count
      num_correct += (preds == y).sum()
      num_correct_adv += (preds_adv == y).sum()
      num_samples += len(preds)

      # Adding probs to dict
      count=0
      for i in idx.numpy():
        score_curr = scores_adv[count].cpu().detach().numpy()
        prob_dict[str(i)] = softmax(score_curr)
        # print(count)
        count+=1

      # Tracking hard point losses and predictions
      idx_matched = idx[~ez].numpy()
      m_matched = m[~ez].numpy()
        
      preds_adv_matched = preds_adv[len(unmatched_x):].cpu().numpy()
      loss_adv_matched = batch_loss_adv[len(unmatched_x):].cpu().detach().numpy()
      y_np_matched = matched_y.cpu().numpy()
      ez_np = np.ones(len(x), dtype=bool)
      ez_np[len(unmatched_x):] = 0

      for i in range(len(y_np_matched)):
        pred_dict[str(idx_matched[i])] = [m_matched[i],y_np_matched[i],preds_adv_matched[i]]

      loss_dict = track_hard_losses(ez_np, batch_loss_adv, batch_loss_ben, 
                          loss_dict, t)
      
      if not training_time:
        if args.viz and steps == 1:
            if not os.path.exists(figure_dir_name):
              os.makedirs(figure_dir_name)
            custom_save_image(adv_x, preds_adv, y, args, figure_dir_name, 
                              train_data)
      if n_batches > 0 and steps==n_batches:
          break
      steps += 1

  acc = float(num_correct) / num_samples
  acc_adv = float(num_correct_adv) / num_samples
  print('Clean accuracy: {:.2f}% ({}/{})'.format(
      100.*acc,
      num_correct,
      num_samples,
  ))
  print('Adversarial accuracy: {:.2f}% ({}/{})'.format(
      100.*acc_adv,
      num_correct_adv,
      num_samples,
  ))

  if args.track_hard:
    if not training_time:
      hard_point_class_count(pred_dict)
    if len(loss_dict['batch_losses_hard'])>0:
      print('Reporting hard losses')
      if training_time:
        json.dump(loss_dict, f)
        f.write('\n')
      print('Adv loss easy: %.8f' % np.mean(loss_dict['batch_losses_easy']))
      print('Adv loss hard: %.8f' % np.mean(loss_dict['batch_losses_hard']))
      print('Ben loss easy: %.8f' % np.mean(loss_dict['batch_losses_ben_easy']))
      print('Ben loss hard: %.8f' % np.mean(loss_dict['batch_losses_ben_hard']))

  return 100.*acc, 100.*acc_adv, np.mean(losses_ben), np.mean(losses_adv), prob_dict


def robust_test(model, loss_fn, loader, args, att_dir, epoch=0, training_output_dir_name=None, 
                figure_dir_name=None, n_batches=0, train_data=False, 
                training_time=False):
    """
    n_batches (int): Number of batches for evaluation.
    """
    model.eval()
    num_correct, num_correct_adv, num_samples = 0, 0, 0
    steps = 1
    losses_adv = []
    losses_ben = []
    prob_dict = {}
    if args.track_hard:
      loss_dict = collections.OrderedDict()
      pred_dict = {}
      if training_time and args.track_hard:
        f_name = training_output_dir_name + 'losses.json'
        f = open(f_name,'a+')
        loss_dict['epoch'] = epoch

    for t, (x, y, idx, ez, m) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad=False)
        if att_dir['targeted']:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
        if 'PGD_linf' in att_dir['attack']:
            adv_x = pgd_attack(model, x, x_var, y_target, att_dir['attack_iter'],
                           att_dir['epsilon'], att_dir['eps_step'], att_dir['clip_min'],
                           att_dir['clip_max'], att_dir['targeted'], att_dir['rand_init'])
        elif 'PGD_l2' in att_dir['attack']:
            adv_x = pgd_l2_attack(model, x, x_var, y_target, att_dir['attack_iter'],
                           att_dir['epsilon'], att_dir['eps_step'], att_dir['clip_min'],
                           att_dir['clip_max'], att_dir['targeted'], att_dir['rand_init'], 
                           att_dir['num_restarts'])
        # Predictions
        scores = model(x.cuda()) 
        _, preds = scores.data.max(1)
        scores_adv = model(adv_x)
        _, preds_adv = scores_adv.data.max(1)
        # Losses
        batch_loss_adv = loss_fn(scores_adv, y)
        loss_adv = torch.mean(batch_loss_adv)
        losses_adv.append(loss_adv.data.cpu().numpy())
        batch_loss_ben = loss_fn(scores, y)
        loss_ben = torch.mean(batch_loss_ben)
        losses_ben.append(loss_ben.data.cpu().numpy())
        # Correct count
        num_correct += (preds == y).sum()
#         print(preds)
#         print(preds_adv)
        num_correct_adv += (preds_adv == y).sum()
        num_samples += len(preds)
        # Adding probs to dict
        count=0
        for i in idx.numpy():
          score_curr = scores_adv[count].cpu().detach().numpy()
          prob_dict[str(i)] = softmax(score_curr)
          # print(count)
          count+=1

        if args.track_hard:
          idx_matched = idx[~ez].numpy()
          m_matched = m[~ez].numpy()
          preds_adv_matched = preds_adv[~ez].cpu().numpy()
          loss_adv_matched = batch_loss_adv[~ez].cpu().detach().numpy()
          y_np_matched = y[~ez].cpu().numpy()
          ez_np = ez.data.cpu().numpy()

          for i in range(len(y_np_matched)):
            pred_dict[str(idx_matched[i])] = [m_matched[i],y_np_matched[i],preds_adv_matched[i]]

          loss_dict = track_hard_losses(ez_np, batch_loss_adv, batch_loss_ben, 
                              loss_dict, t)
        
        if not training_time:
          if args.viz and steps == 1:
              if not os.path.exists(figure_dir_name):
                os.makedirs(figure_dir_name)
              custom_save_image(adv_x, preds_adv, y, args, figure_dir_name, 
                                train_data)
        
        if n_batches > 0 and steps==n_batches:
            break
        steps += 1

    acc = float(num_correct) / num_samples
    acc_adv = float(num_correct_adv) / num_samples
    print('Clean accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
    ))
    print('Adversarial accuracy: {:.2f}% ({}/{})'.format(
        100.*acc_adv,
        num_correct_adv,
        num_samples,
    ))

    if args.track_hard:
      if not training_time:
        print('Counting hard points')
        hard_point_class_count(pred_dict)
      if training_time:
        json.dump(loss_dict, f)
        f.write('\n')
      if len(loss_dict['batch_losses_hard'])>0:
        print('Adv loss easy: %.8f' % np.mean(loss_dict['batch_losses_easy']))
        print('Adv loss hard: %.8f' % np.mean(loss_dict['batch_losses_hard']))
        print('Ben loss easy: %.8f' % np.mean(loss_dict['batch_losses_ben_easy']))
        print('Ben loss hard: %.8f' % np.mean(loss_dict['batch_losses_ben_hard']))

    return 100.*acc, 100.*acc_adv, np.mean(losses_ben), np.mean(losses_adv), prob_dict