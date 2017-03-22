import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.models as models

from model import NvidiaModel, FineTuneModel, IntentionModel
from iloader import ImageList

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(
        models.__dict__[name]))
model_names.append('nvidia')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('title', metavar='TIT', help='title to save checkpoint')
parser.add_argument(
    '--arch',
    '-a',
    metavar='ARCH',
    default='resnet18',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet18)')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=90,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--device', '-d', default=0, type=int, metavar='N', help='GPU device ID')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')

args = parser.parse_args()

best_loss = float('inf')

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.FileHandler('checkpoints/' + args.title + '.txt')
formatter = logging.Formatter('%(asctime)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def main():
    global args, best_loss
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        original_model = models.__dict__[args.arch](pretrained=True)
        # model = FineTuneModel(original_model, args.arch, 1)
        model = IntentionModel(original_model, args.arch)

    else:
        print("=> creating model '{}'".format(args.arch))
        model = NvidiaModel()

    model = torch.nn.DataParallel(model).cuda()

    size = (224, 224)
    if args.arch.startswith('nvidia'):
        size = (200, 66)
    elif args.arch.startswith('inceptionv4'):
        size = (299, 299)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, '.')
    valdir = os.path.join(args.data, '.')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        ImageList(
            traindir,
            'train.csv',
            transforms.Compose([
                #transforms.Scale(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            random_flip=True,
            #random_shift=True,
            #random_brightness=True,
            #random_sharpness=True,
            imsize=size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageList(
            valdir,
            'val.csv',
            transforms.Compose([
                #transforms.Scale(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            random_flip=False,
            imsize=size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.MSELoss().cuda()

    params_f = model.module.features.parameters()
    params_c = model.module.classifier.parameters()
    optimizer_f = torch.optim.Adam(params_f, args.lr * 0.1)
    optimizer_c = torch.optim.Adam(params_c, args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer_f, epoch)
        #adjust_learning_rate(optimizer_c, epoch)

        # train for one epoch
        loss_ = train(train_loader, model, criterion,
                      (optimizer_f, optimizer_c), epoch)

        # evaluate on validation set
        loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_loss,
        }
        filename = 'checkpoints/%s_check_%02d_%f_%f.pth.tar' % (
            args.title, epoch, loss_, time.time())
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'checkpoints/%s_best_%02d_%f_%f.pth.tar'
                            % (args.title, epoch, loss, time.time()))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    outputs = AverageMeter()
    targets = AverageMeter()
    speed = AverageMeter()

    optimizer_f, optimizer_c = optimizer

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, intention) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        intention_val = torch.autograd.Variable(intention)

        # compute output
        output = model(input_var, intention_val)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        outputs.update(torch.mean(output.data), input.size(0))
        targets.update(torch.mean(target), input.size(0))

        #compute gradient and do SGD step
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        #optimizer.zero_grad()
        loss.backward()
        #optimizer.step()
        optimizer_f.step()
        optimizer_c.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        speed.update(args.batch_size / (time.time() - end))
        end = time.time()

        if i % args.print_freq == 0:
            logger.debug('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Speed {speed.val:.3f} ({speed.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Out {output.val:.3f} ({output.avg:.3f})\t'
                         'Target {target.val:.3f} ({target.avg:.3f})\t'.format(
                             epoch,
                             i,
                             len(train_loader),
                             batch_time=batch_time,
                             data_time=data_time,
                             speed=speed,
                             loss=losses,
                             output=outputs,
                             target=targets))

    logger.debug(' * Train Average Loss {loss.avg:.5f}'.format(loss=losses))
    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    outputs = AverageMeter()
    targets = AverageMeter()
    speed = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, intention) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        intention_val = torch.autograd.Variable(intention)

        # compute output
        output = model(input_var, intention_val)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        outputs.update(torch.mean(output.data), input.size(0))
        targets.update(torch.mean(target), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        speed.update(args.batch_size / (time.time() - end))
        end = time.time()

        if i % args.print_freq == 0:
            logger.debug('Test: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Speed {speed.val:.3f} ({speed.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Out {output.val:.3f} ({output.avg:.3f})\t'
                         'Target {target.val:.3f} ({target.avg:.3f})\t'.format(
                             epoch,
                             i,
                             len(val_loader),
                             batch_time=batch_time,
                             speed=speed,
                             loss=losses,
                             output=outputs,
                             target=targets))

    logger.debug(
        ' * Validation Average Loss {loss.avg:.5f}'.format(loss=losses))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = 'checkpoints/%s_checkpoint_%f.pth.tar' % (args.title,
                                                         time.time())
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/%s_model_best_%f.pth.tar' %
                        (args.title, time.time()))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
