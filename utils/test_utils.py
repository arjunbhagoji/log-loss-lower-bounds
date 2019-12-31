from torch.autograd import Variable
import torch
import torchvision

import numpy as np
import time 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack, pgd_l2_attack


def test(model, loader, figure_dir_name):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    steps = 1
    for x, y in loader:
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
    return acc


def robust_test(model, loader, args, figure_dir_name, n_batches=0):
    """
    n_batches (int): Number of batches for evaluation.
    """
    model.eval()
    num_correct, num_correct_adv, num_samples = 0, 0, 0
    steps = 1
    for x, y, z in loader:
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
                           args.new_attack_iter,
                           args.new_epsilon,
                           args.new_eps_step,
                           args.clip_min,
                           args.clip_max, 
                           args.targeted,
                           args.new_rand_init)
        elif 'PGD_l2' in args.attack:
            adv_x = pgd_l2_attack(model,
               x,
               x_var,
               y_target,
               args.new_attack_iter,
               args.new_epsilon,
               args.new_eps_step,
               args.clip_min,
               args.clip_max, 
               args.targeted,
               args.new_rand_init)
        scores = model(x.cuda()) 
        _, preds = scores.data.max(1)
        scores_adv = model(adv_x)
        _, preds_adv = scores_adv.data.max(1)
        num_correct += (preds == y).sum()
        num_correct_adv += (preds_adv == y).sum()
        num_samples += len(preds)
        # if args.viz and steps == 1:
        #     if not os.path.exists(figure_dir_name):
        #       os.makedirs(figure_dir_name)
        #   torchvision.utils.save_image(adv_x, '{}/pert_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
        #                                 args.new_attack_iter, args.new_eps_step))
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

    return acc, acc_adv

def robust_test_during_train(model, loss_fn, loader, args, n_batches=0):
    """
    n_batches (int): Number of batches for evaluation.
    """
    model.eval()
    num_correct, num_correct_adv, num_samples = 0, 0, 0
    steps = 1
    losses = []
    losses_adv = []
    for x, y, z in loader:
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
                           args.epsilon,
                           args.eps_step,
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
               args.epsilon,
               args.eps_step,
               args.clip_min,
               args.clip_max, 
               args.targeted,
               args.rand_init)
        scores = model(x.cuda()) 
        _, preds = scores.data.max(1)
        batch_loss = loss_fn(scores, y)
        loss = torch.mean(batch_loss)
        losses.append(loss.data.cpu().numpy())
        scores_adv = model(adv_x)
        _, preds_adv = scores_adv.data.max(1)
        batch_loss_adv = loss_fn(scores_adv, y)
        loss_adv = torch.mean(batch_loss_adv)
        losses_adv.append(loss_adv.data.cpu().numpy())
        num_correct += (preds == y).sum()
        num_correct_adv += (preds_adv == y).sum()
        num_samples += len(preds)
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

    return 100.*acc, 100.*acc_adv, np.mean(losses), np.mean(losses_adv)