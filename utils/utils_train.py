from os.path import join, isdir, isfile
from os import makedirs
import torch
import numpy as np


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

def adjust_learning_rate(args, optimizer, epoch, lr0):
    #lr = np.logspace(-6, -6, num=args.epochs)[epoch]
    lr = np.logspace(-lr0[0], -lr0[1], num=args.epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, lr, save_path):
    name0 = 'lr' + str(lr[0])+str(lr[1])
    name0 = name0.replace('.','_')
    epo_path = join(save_path, name0)
    if not isdir(epo_path):
        makedirs(epo_path)
    if (epoch+1) % 5 == 0:
        filename=join(epo_path, 'checkpoint{}.pth.tar'.format(epoch+1))
        torch.save(state, filename)