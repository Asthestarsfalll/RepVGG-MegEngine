import argparse
import time

from model import repvgg

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F

from utils import AverageMeter, build_dataset
logging = mge.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Training")
    parser.add_argument("-d", "--data", metavar="DIR",
                        help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="repvggA0",
        help="model architecture (default: RepVGGA0,option:RepVGGA0,RepVGGA1,RepVGGA2,RepVGGB0,RepVGGB1,RepVGGB1g2,RepVGGB1g4,RepVGGB2,RepVGGB2g2,RepVGGB2g4,RepVGGB3,RepVGGB3g2,RepVGGB3g4,RepVGGD2se )",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None, 
        help="path to model checkpoint"
    )
    parser.add_argument(
        "-j", "--workers", 
        default=2, 
        type=int
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--deploy",
        default=0, 
        type=int)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", default=23456, type=int)
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    args = parser.parse_args()

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork("gpu")

    if args.world_size * args.ngpus > 1:
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
    # build dataset
    _, valid_dataloader = build_dataset(args)

    # build model
    model = repvgg.__dict__[args.arch](args.deploy)
    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = mge.load(args.model)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

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

    model.eval()
    _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
    logging.info(
        "Test Acc@1 %.3f, Acc@5 %.3f",
        valid_acc1,
        valid_acc5,
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
