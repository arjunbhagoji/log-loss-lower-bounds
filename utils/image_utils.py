import torch
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from PIL import Image

irange = range

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    print(height, width)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid, height, width, padding

def plot_image_grid(grid, height, width, padding, preds_adv=None, y=None, file_name=None, dpi=224, index=None):
    grid_shape = grid.shape
    # n_rows = len(grid)
    # n_cols = len(grid[0])
    n_rows = int((grid_shape[0] - padding)/height)
    n_cols = int((grid_shape[1]-padding)/width)
    print(n_rows, n_cols)

    plt.clf()
    plt.rc("font", family="sans-serif")

    class_labels = ['dress', 'sneaker']

    plt.figure(figsize = (n_cols, n_rows)) #TODO figsize
    count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows+1, n_cols], loc=[r+1,c])
            curr_im_data = grid[r*height+padding:(r+1)*height+padding, c*width+padding:(c+1)*width+padding]
            curr_im = Image.fromarray(curr_im_data)
            ax.imshow(curr_im, interpolation='none') #TODO. controlled color mapping wrt all grid entries, or individually. make input param
            ax.set_xticks([])
            ax.set_yticks([])

            if preds_adv is not None:
                curr_label = 'Pred:' + class_labels[preds_adv[count]] + '; True:' + class_labels[y[count]]

                # if not r: #column labels
                # if col_labels != []:
                ax.set_title(curr_label,
                             rotation=0.0,
                             horizontalalignment='left',
                             verticalalignment='bottom',
                             fontsize=8
                             )

            if index is not None:
                if count < len(index):
                    curr_label = index[count]
                    ax.set_ylabel(curr_label,
                                 rotation=0.0,
                                 horizontalalignment='left',
                                 verticalalignment='bottom',
                                 fontsize=8
                                 )

            count += 1
                            
                    # print('skip')

            # if not c: #row labels
            #     if row_labels_left != []:
            #         txt_left = [l+'\n' for l in row_labels_left[r]]
            #     ax.set_ylabel(''.join(txt_left),
            #                   rotation=0,
            #                   verticalalignment='center',
            #                   horizontalalignment='right',
            #                   )
            # if c == n_cols-1:
            #     if row_labels_right != []:
            #         txt_right = [l+'\n' for l in row_labels_right[r]]
            #         ax2 = ax.twinx()
            #         ax2.set_xticks([])
            #         ax2.set_yticks([])
            #         ax2.set_ylabel(''.join(txt_right),
            #                       rotation=0,
            #                       verticalalignment='center',
            #                       horizontalalignment='left',
            #                       fontsize=10
            #                        )

    print ('saving figure to {}'.format(file_name))
    plt.savefig(file_name, orientation='landscape', dpi=dpi)


def custom_save_image(adv_x, preds_adv, y, args, figure_dir_name, train_data):
	
	if train_data:
		file_name = '{}/train_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
	                        args.new_attack_iter, args.new_eps_step)
	else:
		file_name = '{}/test_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
	                        args.new_attack_iter, args.new_eps_step)

	torch_grid, height, width, padding = make_grid(adv_x)
	numpy_grid = torch_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
					'cpu', torch.uint8).numpy()
	print(numpy_grid.shape)
	
	preds_adv = preds_adv.cpu().numpy()
	y = y.cpu().numpy()

	plot_image_grid(numpy_grid, height, width, padding, preds_adv, y, file_name=file_name, dpi=224)

def save_image_simple(x, args, figure_dir_name, train_data=None, hard=None, indices=None):
    file_name = '{}/train_first.jpg'.format(figure_dir_name)

    # if train_data:
    #     if hard:
    #         file_name = '{}/train_hard_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
    #                         args.new_attack_iter, args.new_eps_step)
    #     else:
    #         file_name = '{}/train_easy_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
    #                         args.new_attack_iter, args.new_eps_step)
    # else:
    #     if hard:
    #         file_name = '{}/test_hard_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
    #                         args.new_attack_iter, args.new_eps_step)
    #     else:
    #         file_name = '{}/test_easy_{}_{}_{}.jpg'.format(figure_dir_name, args.new_epsilon, 
    #                         args.new_attack_iter, args.new_eps_step)

    torch_grid, height, width, padding = make_grid(x)
    numpy_grid = torch_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
                    'cpu', torch.uint8).numpy()
    print(numpy_grid.shape)
    
    plot_image_grid(numpy_grid, height, width, padding, file_name=file_name, dpi=224, index=indices)