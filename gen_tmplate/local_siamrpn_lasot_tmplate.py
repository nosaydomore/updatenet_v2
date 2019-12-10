import cv2  # imread
import torch
import numpy as np
from os.path import join
import os
from got10k.experiments import ExperimentOTB,ExperimentVOT


from model.SiamRPN import SiamRPNBIG,SiamRPNotb,SiamRPNvot
from tracker.run_SiamRPN_init_upd import SiamRPN_init, SiamRPN_track
from tracker.run_SiamRPN_upd_k import SiamRPN_init_upd, SiamRPN_track_upd
from updatenet.upd_net import UpdateResNet
from utils.utils_rpn import rect_2_cxy_wh, cxy_wh_2_rect, overlap_ratio


net_set={
    'big':SiamRPNBIG,
    'otb':SiamRPNotb,
    'vot':SiamRPNvot
}


def init_lasot():
    video_path = '/media/data1/zph_data/LaSOT'
    list_file = np.load('/media/data1/zph_data/2_tracking/updatenet/selected_cat_videos.npy')
    category = os.listdir(video_path)
    category.sort()
    return list_file, category

def init_video(video_path,tmp_cat,video):
    print(video)
    gt_path = join(video_path, tmp_cat, video, 'groundtruth.txt')
    ground_truth = np.loadtxt(gt_path, delimiter=',')
    # ground_truth=ground_truth.astype(np.int)
    num_frames = len(ground_truth); img_path = join(video_path, tmp_cat, video, 'img');imgFiles = [join(img_path, '%08d.jpg') % i for i in range(1, num_frames + 1)]
    return ground_truth, imgFiles, num_frames

def init_net(is_init, rpn_net_path, upd_net_path, net_type='big'):
    net = net_set[net_type]()
    net.load_state_dict(torch.load(rpn_net_path))
    net.eval().cuda()
    if is_init:
        return net
    else:
        if upd_net_path==None:
            assert 'is_init is False, upd_net_path should be not None'
        upd_net = UpdateResNet()
        upd_net_dict = torch.load(upd_net_path)
        print('loading update_net state_dict...')
        upd_net.load_state_dict(upd_net_dict['state_dict'])
        upd_net.eval().cuda()
        return net, upd_net

def append_template(template_acc,template_cur,template_gt,state,is_init):
    template_acc.append(state['z_f'])
    template_cur.append(state['z_f_cur'])
    if is_init:
        template_gt.append(state['gt_z_f'])

def append_gt(gt, frame, num_frames):
    if frame == (num_frames - 1):  # last frame
        gt.append(0)
    else:
        gt.append(1)

def append_zero_tmp(template_acc,template_cur,template_gt,is_init):
    template_acc.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32))
    template_cur.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32))
    if is_init:
        template_gt.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32))

def init_tmp_path(setfile,step_name='templates_step1_std'):

    temp_base_path = '/media/data1/zph_data/2_tracking/undateNet_template_npy'
    temp_path = os.path.join(temp_base_path, setfile + '_'+step_name)
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)
    return temp_path
def save_template_and_idx(template_acc, template_cur, template_gt, init0, init, pre, gt, temp_path):
    print(len(template_acc),template_acc[0].shape)
    template_acc = np.concatenate(template_acc, axis=0);
    template_cur = np.concatenate(template_cur, axis=0);

    np.save(temp_path + '/template.npy', template_acc);
    np.save(temp_path + '/templatei.npy', template_cur);
    if len(template_gt)!=0:
        template_gt = np.concatenate(template_gt, axis=0)
        np.save(temp_path + '/template0.npy', template_gt)

    np.save(temp_path + '/init0.npy', init0);
    np.save(temp_path + '/init.npy', init);
    np.save(temp_path + '/pre.npy', pre);
    np.save(temp_path + '/gt.npy', gt);


def gen_siamrpn_template(rpn_net_path, upd_net_path, video_path, net_type, is_init=False):

    upd_net = None
    if is_init:
        net = init_net(is_init, rpn_net_path, upd_net_path, net_type)
    else:
        net, upd_net=init_net(is_init, rpn_net_path, upd_net_path, net_type)

    reset = 1; frame_max = 300
    list_file, category = init_lasot()

    template_acc = []; template_cur = []; template_gt = []; init0 = []; init = []; pre = []; gt = []  # init0 is reset init
    for tmp_cat in category:
        videos = os.listdir(join(video_path, tmp_cat)); videos.sort()
        for video in videos:
            if video not in list_file:
                continue
            # if video not in ['bear-15']:
            #     continue
            # print(video)
            ground_truth, imgFiles, num_frames = init_video(video_path,tmp_cat,video)
            frame = 0;
            while frame < num_frames:
                gt_bb = ground_truth[frame]-[1,1,0,0]; gt_pos,gt_sz = rect_2_cxy_wh(gt_bb)
                w,h = gt_sz[0], gt_sz[1]

                if w * h != 0:
                    image_file = imgFiles[frame]
                    # target_pos, target_sz = gt_cxy_wh[:2], gt_cxy_wh[2:]
                    im = cv2.imread(image_file)  # HxWxC
                    if is_init:
                        state = SiamRPN_init(im, gt_pos, gt_sz, net)  # init tracker
                    else:
                        state = SiamRPN_init_upd(im, gt_pos, gt_sz, net)

                    # add template
                    append_template(template_acc, template_cur, template_gt, state, is_init)

                    init0.append(0); init.append(frame); frame_reset = 0; pre.append(0); gt.append(1)
                    while frame < (num_frames - 1):
                        frame = frame + 1; frame_reset = frame_reset + 1
                        image_file = imgFiles[frame]
                        if not image_file:
                            break
                        im = cv2.imread(image_file)  # HxWxC

                        gt_bb = ground_truth[frame] - [1, 1, 0, 0];
                        gt_pos, gt_sz = rect_2_cxy_wh(gt_bb)
                        if is_init:
                            state = SiamRPN_track(state, im, gt_pos, gt_sz)  # track
                        else:
                            state = SiamRPN_track_upd(state, im, upd_net)
                        append_template(template_acc, template_cur, template_gt, state, is_init)

                        init0.append(frame_reset); init.append(frame); pre.append(1);
                        append_gt(gt, frame, num_frames)

                        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                        if reset:
                            gt_rect = np.asarray(ground_truth[frame])
                            iou = overlap_ratio(gt_rect, res)
                            if iou <= 0:
                                print('reset!', 'frame:', frame_reset)
                                break
                else:
                    # zero tmp fills
                    append_zero_tmp(template_acc, template_cur, template_gt, is_init)
                    init0.append(0); init.append(frame); pre.append(1);
                    append_gt(gt, frame, num_frames)
                frame = frame + 1  # skip
            # break
        # break
    return template_acc, template_cur, template_gt, init0, init, pre, gt


if __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    net_base_path = '/media/data1/zph_data/2_tracking/pretrained_model'
    rpn_net_path = join(net_base_path, 'SiamRPNBIG.model')
    upd_net_path = '/media/data1/zph_data/2_tracking/updatenet_train_model/work_siamrpn_upd_k2/lr67/checkpoint50.pth.tar'
    video_path = '/media/data1/zph_data/LaSOT'
    setfile = 'update_set1'
    # temp_path = init_tmp_path(setfile)
    temp_path = init_tmp_path(setfile, step_name='templates_step3_std')
    template_acc, template_cur, template_gt, init0, init, pre, gt=gen_siamrpn_template(rpn_net_path, upd_net_path, video_path, 'big', is_init=False)

    save_template_and_idx(template_acc, template_cur, template_gt, init0, init, pre, gt, temp_path)







