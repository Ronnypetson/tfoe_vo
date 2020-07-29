import os
import sys
import time
import numpy as np
import cv2
import torch
from liegroups import SE3
#from matplotlib import pyplot as plt
from opt_local import OptSingle
import pykitti
from versions import files_to_hash, save_state
from utils import norm_t, plot_trajs, ba_graph
from utils import save_poses, pt_cloud, plot_pt_cloud
from reproj import triangulate, rel_scale


class KpT0_BA:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self, h, w, basedir, seq, fast_th=25):
        super().__init__()
        self.size = (h, w)
        self.kitti = pykitti.odometry(basedir, seq)
        self.gt_odom = self.kitti.poses
        self.seq_len = len(self.gt_odom)
        self.camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                       [0.0, 718.8560, 185.2157],
                                       [0.0, 0.0,      1.0]])
        self.c_ = np.linalg.inv(self.camera_matrix)
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=fast_th,
                                                               nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._kpts = {} # keypoint cache, i -> kp
        self._kptsij = {} # [i,j] -> kpij
        self._flow = {} # [i,j] -> f
        self._T0 = {} # i -> T
        self._ep0 = {} # i -> ep
        self._Tgt = {} # i -> Tgt
        self._gTgt = {} # i -> gTgt
        self._vids = {} # [i,j] -> vid
        self._avids = {} # [i,j] -> avid
        self._moving = {} # i -> moving
        self._Tij0 = {} # [i,j] -> Tij

    def init_frame(self, i):
        if i not in self._T0:
            i_ = min(i+1, self.seq_len-1)
            im0 = self.kitti.get_cam0(i)
            im0 = np.array(im0)
            im1 = self.kitti.get_cam0(i_)
            im1 = np.array(im1)
            kp0 = self.feature_detector.detect(im0, None)
            kp0 = np.array([[x.pt] for x in kp0], dtype=np.float32)

            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(im0, im1, kp0,
                                                       None, **self.lk_params)
                E, mask = cv2.findEssentialMat(p1, kp0, self.camera_matrix,
                                               cv2.RANSAC, 0.999, 0.1, None)
                vids = [j for j in range(len(mask)) if mask[j] == 1.0]
                avids = vids

                _, R, t, mask = cv2.recoverPose(E, p1, kp0, self.camera_matrix, mask=mask)
                T0 = np.eye(4)
                T0[:3, :3] = R
                T0[:3, 3:] = t
                #T0[:3, 3:] = t / (t[-1] + 1e-10)

                if len(mask) > 3:
                    vids_ = [j for j in range(len(mask)) if mask[j] == 1.0]
                    if len(vids_) > 3:
                        vids = vids_
                if np.mean(np.abs(p1[vids] - kp0[vids])) < 1e-2:
                    T0 = np.eye(4)
                    moving = False
                else:
                    moving = True
                ep2 = T0[:3, 3:]
                ep2 = ep2 / (ep2[-1] + 1e-8)
                ep2 = self.camera_matrix @ ep2
                ep0 = ep2[:2, 0]

                inert = np.linalg.inv(self.gt_odom[i])
                Tgt = inert @ self.gt_odom[i_]

                #norm_gt = np.linalg.norm(Tgt[:3, 3:])
                #T0 = norm_t(T0.copy(), norm_gt)

                self._kpts[i] = kp0
                self._flow[(i, i_)] = p1 - kp0
                self._T0[i] = T0
                self._Tij0[(i, i_)] = T0
                self._ep0[i] = ep0
                self._Tgt[i] = Tgt
                if i > 0:
                    self._gTgt[i] = T0 @ self._gTgt[i-1] #self.gt_odom[i_]
                else:
                    self._gTgt[i] = T0
                self._vids[(i, i_)] = vids
                self._avids[(i, i_)] = avids
                self._moving[i] = moving
            except Exception as e:
                print(e)
                raise e

    def init_BA(self, i, j):
        msg = f'Check frame indexes: {i} {j}'
        assert 0 <= i < self.seq_len, msg
        assert 0 <= j < self.seq_len, msg
        assert i != j, msg

        self.init_frame(i)
        self.init_frame(j)
        kp0 = self._kpts[i]
        if (i, j) not in self._flow:
            im_i = self.kitti.get_cam0(i)
            im_i = np.array(im_i)
            im_j = self.kitti.get_cam0(j)
            im_j = np.array(im_j)

            #kp0 = self.feature_detector.detect(im_i, None)
            #kp0 = np.array([[x.pt] for x in kp0], dtype=np.float32)

            kp1, st, err = cv2.calcOpticalFlowPyrLK(im_i, im_j, kp0,
                                                   None, **self.lk_params)

            E, mask = cv2.findEssentialMat(kp1, kp0, self.camera_matrix,
                                           cv2.RANSAC, 0.999, 0.1, None)

            vids = [k for k in range(len(mask)) if mask[k] == 1.0]
            avids = vids

            _, R, t, mask = cv2.recoverPose(E, kp1, kp0, self.camera_matrix, mask=mask)
            T0 = np.eye(4)
            T0[:3, :3] = R
            T0[:3, 3:] = t
            #T0[:3, 3:] = t / (t[-1] + 1e-10)

            if len(mask) > 3:
                vids_ = [k for k in range(len(mask)) if mask[k] == 1.0]
                if len(vids_) > 3:
                    vids = vids_

            #self._kptsij[(i, j)] = kp0
            self._flow[(i, j)] = kp1 - kp0
            self._vids[(i, j)] = vids
            self._avids[(i, j)] = avids
            self._Tij0[(i, j)] = T0
        return kp0, self._flow[(i, j)]


def main():
    seq_id = sys.argv[1]
    run_tag = sys.argv[2]
    exp_dir = sys.argv[3]

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    run_date = time.asctime().replace(' ', '_')
    state_fns = ['kitti_local.py', 'opt_local.py', 'utils.py', 'reproj.py']
    run_hash = files_to_hash(state_fns)
    run_dir = f'odom/{run_hash}/{run_date}/'
    save_state('odom/', state_fns)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    bdir = '/home/ronnypetson/Downloads/kitti_seq/dataset/'
    h, w = 376, 1241
    kp = KpT0_BA(h, w, bdir, seq_id, fast_th=15)
    c = kp.camera_matrix
    c_ = np.linalg.inv(c)
    failure_eps = 1e-7
    poses = []
    poses_gt = []
    poses_ = []
    show_cloud = False
    pose0 = np.eye(4)
    pose0_gt = np.eye(4)
    pose0_init = np.eye(4)
    W_poses = []
    W_poses_gt = []
    W_poses_init = []
    cloud_all = np.zeros((3, 1))
    gT = np.zeros((kp.seq_len+1, 6))
    ge = np.zeros((kp.seq_len+1, 2))
    gs = np.ones((kp.seq_len+1, 1))
    ge[0] = np.array([607.1928, 185.2157]) / 1e3
    baw = 4

    try:
        for i in range(0, kp.seq_len, baw - 1):
            kp.init_frame(i)
            Tgt = kp._Tgt[i]
            T = kp._T0[i]

            g = ba_graph(i, i + (baw - 1))
            p = {}
            f = {}
            for ij in g:
                kp.init_BA(ij[0], ij[1])
                p[ij] = kp._kpts[ij[0]]
                f[ij] = kp._flow[ij]

            normT0 = np.linalg.norm(kp._Tgt[i][:3, 3])
            for j in range(baw - 1):
                normT = np.linalg.norm(kp._Tgt[i + j][:3, 3])
                poses.append(norm_t(kp._T0[i + j].copy(), normT))
                poses_gt.append(kp._Tgt[i + j])

            x = {}
            x_ = {}
            for ij in g:
                x[ij] = p[ij][kp._vids[ij], 0, :].transpose(1, 0) # kp._vids[ij]
                z = np.ones((1, x[ij].shape[-1]))
                x[ij] = np.concatenate([x[ij], z], axis=0)

                x_[ij] = (p[ij] + f[ij])[kp._vids[ij], 0, :].transpose(1, 0) # kp._vids[ij]
                x_[ij] = np.concatenate([x_[ij], z], axis=0)

            opt = OptSingle(x, x_, kp._Tij0, c, g) # kp.E
            scale_gt = []

            for j in range(baw):
                T0 = SE3.from_matrix(kp._T0[i + j], normalize=True).log()
                norm_gt = np.linalg.norm(kp._Tgt[i + j][:3, 3])
                sc = norm_gt / normT0
                scale_gt.append(sc)
                gT[i + j] = T0.copy()

            for j in range(baw):
                ge[i + j] = kp._ep0[i + j] / 1e3

            gs[i] = 1.0
            rs0 = 1.0
            for j in range(i + 1, i + baw - 1, 1):
                id0 = j - 1
                id1 = j
                id2 = j + 1
                n12 = np.linalg.norm(f[(id1, id2)], axis=-1)
                n01 = np.linalg.norm(f[(id0, id1)], axis=-1)
                rs = (np.median(n12) / np.median(n01))**(1.0/2.5)
                rs0 = rs * rs0
                gs[j] = rs0
            print(scale_gt)
            print(gs[i: i + baw].T)
            input()

            Tfoe = opt.optimize(gT[i:i + baw],
                                ge[i:i + baw],
                                gs[i:i + baw],
                                freeze=False)
            print('loss', opt.min_obj)

            Tfoe = Tfoe.reshape(-1, 9)
            T_ = Tfoe[0, :6]
            foe = Tfoe[0, 6:8]
            sc = Tfoe[:, 8]

            #gT[i] = T_.copy()
            gT[i:i + baw] = Tfoe[:, :6]
            ge[i:i + baw] = Tfoe[:, 6:8]
            #gs[i:i + baw] = Tfoe[:, 8:]
            #gT[0] = Tfoe[0, :6]
            #ge[0] = Tfoe[0, 6:8]

            print('ep', foe)
            print('scale\t', sc[:-1])
            print('scalegt\t', scale_gt[:-1])

            for j in range(baw - 1):
                #T_ = SE3.exp(T_).as_matrix() # .inv()
                T_ = SE3.exp(Tfoe[j, :6]).as_matrix()
                normT = np.linalg.norm(kp._Tgt[i + j][:3, 3])
                poses_.append(norm_t(T_.copy(), normT))
                #poses_.append(T_.copy())
                #pose0 = pose0 @ T_
                pose0 = pose0 @ norm_t(T_.copy(), normT)
                pose0_gt = pose0_gt @ kp._Tgt[i + j]
                pose0_init = pose0_init @ norm_t(kp._T0[i + j].copy(), normT)
                W_poses.append(pose0)
                W_poses_gt.append(pose0_gt)
                W_poses_init.append(pose0_init)

            if i % baw == baw - 1:
                P = [poses_gt, poses, poses_]
                #P = [poses_gt, poses, W_poses]
                plot_trajs(P, f'{run_dir}/{seq_id}.svg', glb=False)

            if show_cloud:
                scale = normT / (np.linalg.norm(T_[:3, 3])+1e-8)
                p = torch.from_numpy(p[kp._vids[i], 0]).double()
                p_ = p + torch.from_numpy(f[kp._vids[i], 0]).double()
                T_acc = torch.from_numpy(pose0).double()
                foe = torch.from_numpy(foe).double().unsqueeze(-1)
                c_tc = torch.from_numpy(c).double()
                cloud = pt_cloud(p, p_, T_acc, foe, scale, c_tc, T_)
                cloud_all = np.concatenate([cloud_all, cloud], axis=1) # [:,:20]

            if i % 10 == 9 and show_cloud:
                plot_pt_cloud(np.array(cloud_all), f'{run_dir}/{seq_id}_pt_cloud.svg')

        save_poses(W_poses, f'{run_dir}/KITTI_{seq_id}.txt')
        save_poses(W_poses, f'{exp_dir}/KITTI_{seq_id}_{run_tag}.txt')
        save_poses(W_poses_gt, f'{exp_dir}/KITTI_{seq_id}_{run_tag}_gt.txt')
        save_poses(W_poses_init, f'{exp_dir}/KITTI_{seq_id}_{run_tag}_init.txt')
    except KeyboardInterrupt as e:
        save_poses(W_poses, f'{run_dir}/KITTI_{seq_id}.txt')
        save_poses(W_poses, f'{exp_dir}/KITTI_{seq_id}_{run_tag}.txt')
        save_poses(W_poses_gt, f'{exp_dir}/KITTI_{seq_id}_{run_tag}_gt.txt')
        save_poses(W_poses_init, f'{exp_dir}/KITTI_{seq_id}_{run_tag}_init.txt')
        raise e


if __name__ == '__main__':
    main()
