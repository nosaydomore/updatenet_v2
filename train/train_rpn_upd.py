import argparse
from os.path import join, isdir, isfile
from os import makedirs

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from updatenet.upd_net import UpdateResNet

from utils.utils_train import AverageMeter,adjust_learning_rate,save_checkpoint

def load_data(tem0_path,tem_path):
    print('load_data')
    dataram = dict()

    print('loading template_gt....')
    dataram['template0'] = np.load(join(tem0_path, 'template0.npy'))  # template_gt
    print('loading template_acc....')
    dataram['template'] = np.load(join(tem_path, 'template.npy'))  # template_acc
    print('loading template_cur....')
    dataram['templatei'] = np.load(join(tem_path, 'templatei.npy'))  # template_cur

    dataram['pre'] = np.load(join(tem_path, 'pre.npy'))
    dataram['gt'] = np.load(join(tem_path, 'gt.npy'))
    dataram['init0'] = np.load(join(tem_path, 'init0.npy'))
    dataram['train'] = np.arange(len(dataram['gt']), dtype=np.int)

    return dataram

def train(args,dataram,model,save_path):
    # lrs = np.array(
    #     [[4, 6], [4, 7], [4.5, 5], [4.5, 6], [4.5, 7], [5, 5], [5, 6], [5, 7], [5, 8], [6, 6], [6, 7], [6, 8], [7, 7],
    #      [7, 8], [6.5, 6.5], [6.5, 7], [6.5, 8], [7, 9], [7, 10], [8, 8], [8, 9], [9, 9], [9, 10], [10, 10]])
    lrs = np.array([[6, 7]])
    for ii in np.arange(0, lrs.shape[0]):
        print('start training lr_state:', lrs[ii])
        # model = UpdateResNet()
        model.cuda()
        criterion = nn.MSELoss(size_average=False).cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        for epoch in range(args.start_epoch, args.epochs):
            # glo_step += 1
            adjust_learning_rate(args, optimizer, epoch, lrs[ii])
            losses = AverageMeter()
            # subset = shuffle(subset)
            subset = np.random.permutation(dataram['train'])
            for t in range(0, len(subset), args.batch_size):
                batchStart = t
                batchEnd = min(t + args.batch_size, len(subset))
                batch = subset[batchStart:batchEnd]
                init_index = dataram['init0'][
                    batch]  # format: [0,1,2,3,4,5,0,1,2,3,4,5] the elements denote the distance between current frame and begin frame.
                pre_index = dataram['pre'][batch]  # [0,1,1,1,1,1]
                gt_index = dataram['gt'][batch]

                # reset diff T0
                for rr in range(len(init_index)):
                    if init_index[rr] != 0:
                        init_index[rr] = np.random.choice(init_index[rr], 1)

                cur = dataram['templatei'][batch]  # current frame template
                init = dataram['template0'][batch - init_index]  # the first frame GT template in a video sequence
                pre = dataram['template'][
                    batch - pre_index]  # to ensure the acculmate_template belong to previous frame
                # gt = dataram['template0'][batch+gt_index-1]
                gt = dataram['template0'][batch + gt_index]
                # pdb.set_trace()
                temp = np.concatenate((init, pre, cur), axis=1)
                input_up = torch.Tensor(temp)
                target = torch.Tensor(gt)
                init_inp = Variable(torch.Tensor(init)).cuda()  # T_gt_0
                input_up = Variable(input_up).cuda()
                target = Variable(target).cuda()
                # compute output
                output = model(input_up, init_inp)
                loss = criterion(output, target) / target.size(0)

                # measure accuracy and record loss
                # print(loss.cpu().data.numpy())

                losses.update(loss.cpu().data.numpy())
                # writer.add_scalar('Train/avg_loss', losses.avg, glo_step)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if t % args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        str(epoch).zfill(2), str(t).zfill(5), len(subset), loss=losses))
            save_checkpoint({'state_dict': model.state_dict()}, epoch, lrs[ii],save_path)
def main():
    parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
    parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
    parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
    parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-5)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', '-s', default='./work_step1_std_0_0_curisgt', type=str, help='directory for saving')

    args = parser.parse_args()

    print(args)
    best_loss = 1e6
    tem0_path = '/media/data1/zph_data/2_tracking/undateNet_template_npy/update_set1_templates_step1_std'
    tem_path = '/media/data1/zph_data/2_tracking/undateNet_template_npy/update_set1_templates_step3_std'
    save_path = '/media/data1/zph_data/2_tracking/updatenet_train_model/work_siamrpn_upd_k2'

    dataram = load_data(tem0_path,tem_path)
    model = UpdateResNet()
    upd_weight_path='/media/data1/zph_data/2_tracking/updatenet_train_model/work_siamrpn_upd_k2/lr67/checkpoint50.pth.tar'
    upd_model_weight=torch.load(upd_weight_path)['state_dict']
    model.load_state_dict(upd_model_weight)
    train(args, dataram, model, save_path)



if __name__ == '__main__':
    cudnn.benchmark = True
    main()

