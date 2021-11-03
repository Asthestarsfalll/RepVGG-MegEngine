import argparse
import os
from model import repvgg
import bisect
import time

import megengine as mge
import megengine.functional as F
import megengine.distributed as dist

import megengine.autodiff as autodiff
from utils import AverageMeter, ProgressMeter, build_dataset, sgd_optimizer, CosineAnnealingLR


logging = mge.logger.get_logger()

best_acc1 = 0
best_acc5 = 0


def main():
    parser = argparse.ArgumentParser(description="MegEngine RepVGG Training")
    parser.add_argument("-d", "--data", metavar="DIR",
                        help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="repvggA0",
        help="model architecture (default: RepVGGA0,option:RepVGGA0,RepVGGA1,RepVGGA2,RepVGGB0,RepVGGB1,RepVGGB1g2,RepVGGB1g4,RepVGGB2,RepVGGB2g2,RepVGGB2g4,RepVGGB3,RepVGGB3g2,RepVGGB3g4,RepVGGD2se)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="output",
        help="path to save checkpoint and log",
    )
    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="number of total epochs to run (default: 90)",
    )
    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=256,
        type=int,
        help="batch size for single GPU (default: 64)",
    )
    parser.add_argument(
        '--val-batch-size', '--vbs',
        default=256,
        type=int
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        metavar="LR",
        default=0.1,
        type=float,
        help="learning rate for single GPU (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        help="weight decay (default: 1e-4)"
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)'
    )
    parser.add_argument(
        "-j", "--workers",
        default=2,
        type=int
    )
    parser.add_argument(
        "-p", "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 20)"
    )
    parser.add_argument(
        '--custwd',
        dest='custwd',
        action='store_true',
        help='Use custom weight decay. It improves the accuracy and makes quantization easier.')
    parser.add_argument(
        "--dist-addr",
        default="localhost"
    )
    parser.add_argument(
        "--dist-port",
        default=23456,
        type=int
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int
    )
    parser.add_argument(
        '--tag',
        default='test',
        type=str,
        help='the tag for identifying the log and model files. Just a string.'
    )

    args = parser.parse_args()

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork("gpu")

    if args.world_size * args.ngpus > 1:
        print("Use GPU: {} for training".format(args.ngpus))
        dist_worker = dist.launcher(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=args.world_size * args.ngpus,
            rank_start=args.rank * args.ngpus,
            n_gpus=args.ngpus
        )(worker)
        dist_worker(args)
    else:
        worker(args)


def worker(args):
    # pylint: disable=too-many-statements
    if dist.get_rank() == 0:
        os.makedirs(os.path.join(args.save, args.arch), exist_ok=True)
        mge.logger.set_log_file(
            os.path.join(args.save, args.arch, args.tag+"log.txt"))

    # build dataset
    train_dataloader, valid_dataloader = build_dataset(args)
    train_queue = iter(train_dataloader)  # infinite
    steps_per_epoch = 1281167 // (dist.get_world_size() * args.batch_size)

    # Optimizer
    opt = sgd_optimizer(model, args.lr*dist.get_world_size(),
                        args.momentum, args.weight_decay, args.custwd)
    cos_opt = CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=0)  # change lr every step

    # build model
    model = repvgg.__dict__[args.arch]()

    if args.resume and dist.get_rank() == 0:
        if os.path.isfile(args.resume):
            global best_acc1, best_acc5
            logging.info("=> loading checkpoint '%s'", args.resume)
            checkpoint = mge.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc5 = checkpoint['best_acc5']
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            cos_opt.load_state_dict(checkpoint['scheduler'])
            logging.info("=> loaded checkpoint '%s' (epoch %d)",
                         args.resume, checkpoint['epoch'])
        else:
            logging.info("=> no checkpoint found at '%s'", args.resume)

    # Sync tensor and buffers
    if dist.get_world_size() > 1:
        print(f'world_size is{dist.get_world_size()}')
        dist.bcast_list_(model.tensor())
        dist.bcast_list_(model.buffers())

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb(
            "mean") if dist.get_world_size() > 1 else None,
    )

    # train and valid func
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label)
            acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
            gm.backward(loss)
            opt.step().clear_grad()
        return loss, acc1, acc5

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
        # calculate mean values
        if dist.get_world_size() > 1:
            loss = F.distributed.all_reduce_sum(loss) / dist.get_world_size()
            acc1 = F.distributed.all_reduce_sum(acc1) / dist.get_world_size()
            acc5 = F.distributed.all_reduce_sum(acc5) / dist.get_world_size()
        return loss, acc1, acc5

    # start training
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    for step in range(0, args.epochs * steps_per_epoch):
        lr = adjust_learning_rate(step)

        t = time.time()

        image, label = next(train_queue)
        image = mge.tensor(image, dtype="float32")
        label = mge.tensor(label, dtype="int32")

        loss, acc1, acc5 = train_step(image, label)

        objs.update(loss.item())
        top1.update(100 * acc1.item())
        top5.update(100 * acc5.item())
        clck.update(time.time() - t)

        cos_opt.step()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "Epoch %d Step %d, LR %.4f, %s %s %s %s",
                step // steps_per_epoch,
                step,
                lr,
                objs,
                top1,
                top5,
                clck,
            )
            objs.reset()
            top1.reset()
            top5.reset()
            clck.reset()

        if (step + 1) % steps_per_epoch == 0:
            model.eval()
            _, valid_acc1, valid_acc5 = valid(
                valid_step, valid_dataloader, args)
            model.train()
            logging.info(
                "Epoch %d Test Acc@1 %.3f, Acc@5 %.3f",
                (step + 1) // steps_per_epoch,
                valid_acc1,
                valid_acc5,
            )
            if valid_acc1 > best_acc1:
                best_acc1 = valid_acc1
            if valid_acc5 > best_acc5:
                best_acc5 = valid_acc5

            if dist.get_rank() == 0:
                mge.save(
                    {
                        "epoch": (step + 1) // steps_per_epoch,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": cos_opt.state_dict(),
                        'best_acc1': best_acc1,
                        'best_acc5': best_acc5
                    },
                    os.path.join(args.save, args.arch, "checkpoint.pkl"),
                )


def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        image = mge.tensor(image, dtype="float32")
        label = mge.tensor(label, dtype="int32")

        n = image.shape[0]

        loss, acc1, acc5 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top5.update(100 * acc5.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info("Test step %d, %s %s %s %s",
                         step, objs, top1, top5, clck)

    return objs.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
