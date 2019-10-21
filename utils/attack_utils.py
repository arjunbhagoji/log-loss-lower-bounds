import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import numpy as np

def cal_loss(y_out, y_true, targeted):
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(y_out, y_true)
    if targeted:
        return loss_cal
    else:
        return -1*loss_cal

def generate_target_label_tensor(true_label, args):
    t = torch.floor(args.n_classes*torch.rand(true_label.shape)).type(torch.int64)
    m = t == true_label
    t[m] = (t[m]+ torch.ceil((args.n_classes-1)*torch.rand(t[m].shape)).type(torch.int64)) % args.n_classes
    return t

def pgd_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal = cal_loss(output, tar_label_variable, targeted)
        loss_cal.backward()
        x_grad = -1 * eps_step * torch.sign(img_variable.grad.data)
        adv_temp = img_variable.data + x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps_max, eps_max)
        x_adv = image_tensor + total_grad
        x_adv = torch.clamp(torch.clamp(
            x_adv-image_tensor, -1*eps_max, eps_max)+image_tensor, clip_min, clip_max)
        img_variable.data = x_adv
    #print("peturbation= {}".format(
    #    np.max(np.abs(np.array(x_adv)-np.array(image_tensor)))))
    return img_variable

def pgd_l2_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal = cal_loss(output, tar_label_variable, targeted)
        loss_cal.backward()
        raw_grad = img_variable.grad.data
        grad_norm = torch.max(
               raw_grad.view(raw_grad.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
        grad_dir = raw_grad/grad_norm.view(raw_grad.size(0),1,1,1)
        adv_temp = img_variable.data +  -1 * eps_step * grad_dir
        # Clipping total perturbation
        total_grad = adv_temp - image_tensor
        total_grad_norm = torch.max(
               total_grad.view(total_grad.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
        total_grad_dir = total_grad/total_grad_norm.view(total_grad.size(0),1,1,1)
        total_grad_norm_rescale = torch.min(total_grad_norm, torch.tensor(eps_max).cuda())
        clipped_grad = total_grad_norm_rescale.view(total_grad.size(0),1,1,1) * total_grad_dir
        x_adv = image_tensor + clipped_grad
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
        img_variable.data = x_adv
    # print("peturbation= {}".format(
    #    np.max(np.abs(np.array(x_adv)-np.array(image_tensor.data)))))
    return img_variable