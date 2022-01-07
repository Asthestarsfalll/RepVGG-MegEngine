import math

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.optimizer as optim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_train_trans(args):
    normalize = T.Normalize(
        mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]  # BGR
    )
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(0.5),
            normalize,
            T.ToMode("CHW")
        ])
    else:
        raise ValueError('Not yet implemented.')
    return trans


def get_val_trans(args):
    normalize = T.Normalize(
        mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]  # BGR
    )
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            normalize,
            T.ToMode("CHW")
        ])
    else:
        trans = T.Compose([
            T.Resize(
                args.resolution, interpolation=cv2.INTER_LINEAR),
            T.CenterCrop(args.resolution),
            normalize,
            T.ToMode("CHW")
        ])
    return trans


def build_dataset(args, is_train=True):
    if is_train:
        train_trans = get_train_trans(args)
        train_dataset = data.dataset.ImageNet(args.data, train=True)
        train_sampler = data.Infinite(
            data.RandomSampler(
                train_dataset, batch_size=args.batch_size, drop_last=True)
        )
        train_dataloader = data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            transform=train_trans,
            num_workers=args.workers
        )
    else:
        train_dataloader = None
    
    val_trans = get_val_trans(args)
    valid_dataset = data.dataset.ImageNet(args.data, train=False)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=args.val_batch_size, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=val_trans,
        num_workers=args.workers
    )
    return train_dataloader, valid_dataloader


def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        # if not value.requires_grad:
        #     continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('dense' in key or 'pointwise' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            # Just a Caffe-style common practice. Made no difference.
            apply_lr = 2 * lr
        params += [{'params': [value], 'lr': apply_lr,
                    'weight_decay': apply_weight_decay}]
    optimizer = optim.SGD(params, lr, momentum=momentum)
    return optimizer


class CosineAnnealingLR(optim.LRScheduler):
    """
    A Simple Implement Of CosineAnnealingLR (https://arxiv.org/abs/1608.03983)
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        current_epoch (int): the index of current epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min, current_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, current_epoch)

    def get_lr(self):
        if self.current_epoch == -1:
            return self.base_lrs
        elif (self.current_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.current_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.current_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


if __name__ == '__main__':
    # *****************  test CosineAnnealingLR  ***************************
    import matplotlib.pyplot as plt
    import megengine.module as M
    class Conv(M.Module):
        def __init__(self):
            super(Conv, self).__init__()
            self.layer = M.Conv2d(3,3,1)
        def forward(self, inputs):
            return self.layer(inputs)
    model = Conv()
    opt = mge.optimizer.SGD(model.parameters(), 0.1)
    cos_opt = CosineAnnealingLR(opt, T_max=2000, eta_min=0)
    lr_list = []
    for i in range(2000):
        opt.step()
        cos_opt.step()
        cur_lr = opt.param_groups[-1]['lr']
        lr_list.append(cur_lr)
    x_list = list(range(len(lr_list)))
    plt.plot(x_list, lr_list)
    plt.show()
    mge.save(cos_opt.state_dict(), './ckpt/cos.kpl')
    new = CosineAnnealingLR(opt, T_max=2000, eta_min=0)
    new.load_state_dict(mge.load('./ckpt/cos.kpl'))
    print(new.current_epoch)
