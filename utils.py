import megengine.data.transform as T
import megengine.data as data
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


def build_dataset(args, is_train=True):
    normalize = T.Normalize(
        mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
    )
    train_dataset = data.dataset.ImageNet(args.data, train=True)
    if is_train:
        train_sampler = data.Infinite(
            data.RandomSampler(
                train_dataset, batch_size=args.batch_size, drop_last=True)
        )
        train_dataloader = data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            transform=T.Compose(
                [  # Baseline Augmentation for small models
                    T.Resize(256),
                    T.CenterCrop(224),
                    normalize,
                    T.ToMode("CHW"),
                ]
            )
            if (not hasattr(args, 'resolution')) or args.resolution == 224
            else T.Compose(
                [  # Facebook Augmentation for large models
                    T.Resize(args.resolution),
                    T.CenterCrop(args.resolution),
                    normalize,
                    T.ToMode("CHW"),
                ]
            ),
            num_workers=args.workers,
        )
    else:
        train_dataloader = None
    valid_dataset = data.dataset.ImageNet(args.data, train=False)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=args.val_batch_size, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                normalize,
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    return train_dataloader, valid_dataloader


def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
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
