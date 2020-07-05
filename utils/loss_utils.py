import torch.nn as nn

def KL_loss_flat(p_hat, p_opt, y, t):
	loss_function = nn.KLDivLoss(reduction='none')
	# KL divergence function expects log-softmax for the output and softmax for target
	KL_loss = loss_function(p_hat, p_opt)
	count = 0
	# Recovering the actual prob. values
	softmax_fn = nn.Softmax(dim=1)
	p_hat_proper = softmax_fn(p_hat)
	# print(p_hat_proper, p_opt)
	for i, item in enumerate(y):
		if y[i]==0:
			if p_hat_proper[i][0]>p_opt[i][0]:
				KL_loss[i]=0.0
				count+=1
		elif y[i]==1:
			if p_hat_proper[i][1]>p_opt[i][1]:
				KL_loss[i]=0.0
				count+=1
	return KL_loss