

import logging
import time
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import utils.datasets
import utils.utils
from utils.model import CNN
from nni.nas.pytorch.utils import AverageMeter
from nni.retiarii import fixed_arch

logger = logging.getLogger('ECG-NAS')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def train(config, train_loader, model, optimizer, criterion, epoch):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(x)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        accuracy = utils.utils.accuracy(logits, y, topk=(1, 2))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc2"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", accuracy["acc2"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(config, valid_loader, model, criterion, epoch, cur_step):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            accuracy = utils.accuracy(logits, y, topk=(1, 2))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc2"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--log-frequency", default=30, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="/home/xun/code/git/ECG-NAS/examples/nas/oneshot/darts/checkpoint_1ch_labeled_3class.json")

    args = parser.parse_args()
    args.batch_size=10
    print("args.batch_size=",args.batch_size)
    dataset_train, dataset_valid = utils.datasets.get_ECG_data()
    #dataset_train, dataset_valid = datasets.get_dataset("cifar10", cutout_length=16)

    with fixed_arch(args.arc_checkpoint):
        model = CNN(100, 1, 36, 3, args.layers, auxiliary=True)
    criterion = nn.CrossEntropyLoss()
    print("test1")
    model.to(device)
    criterion.to(device)
    print("test2")
    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)
    print("test3")
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    best_top1 = 0.
    for epoch in range(args.epochs):
        drop_prob = args.drop_path_prob * epoch / args.epochs
        model.drop_path_prob(drop_prob)

        # training
        train(args, train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(args, valid_loader, model, criterion, epoch, cur_step)
        best_top1 = max(best_top1, top1)
        lr_scheduler.step()

    EPOCH = 4
    PATH = "model_randsearch.pt"
    LOSS = 0.4
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, PATH)  
