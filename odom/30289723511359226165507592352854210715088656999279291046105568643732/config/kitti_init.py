import os
import sys
import time
import numpy as np
import cv2
import torch
from liegroups import SE3
#from matplotlib import pyplot as plt
from opt_ba import OptSingle
import pykitti
from versions import files_to_hash, save_state
from utils import norm_t, plot_trajs, ba_graph
from utils import save_poses, pt_cloud, plot_pt_cloud


class KpT0_BA:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self, h, w, basedir, seq):
        super().__init__()
        self.size = (h, w)
        self.kitti = pykitti.odometry(basedir, seq)
        self.gt_odom = self.kitti.poses
        self.seq_len = len(self.gt_odom)
        self.camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                       [0.0, 718.8560, 185.2157],
                                       [0.0, 0.0,      1.0]])
        self.c_ = np.linalg.inv(self.camera_matrix)
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                               nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._kpts = {} # keypoint cache, i -> kp
        self._flow = {} # [i,j] -> f
        self._T0 = {} # i -> T
        self._ep0 = {} # i -> ep
        self._Tgt = {} # i -> Tgt
        self._vids = {} # i -> vid
        self._avids = {} # i -> avid
        self._moving = {} # i -> moving

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

                self._kpts[i] = kp0
                self._flow[(i, i_)] = p1 - kp0
                self._T0[i] = T0
                self._ep0[i] = ep0
                self._Tgt[i] = Tgt
                self._vids[i] = vids
                self._avids[i] = avids
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
            kp1, st, err = cv2.calcOpticalFlowPyrLK(im_i, im_j, kp0,
                                                   None, **self.lk_params)
            self._flow[(i, j)] = kp1 - kp0
        return kp0, self._flow[(i, j)]


def main():
    seq_id = sys.argv[1]
    run_date = time.asctime().replace(' ', '_')
    state_fns = ['kitti_init.py', 'opt.py', 'utils.py', 'reproj.py', 'opt.py']
    run_hash = files_to_hash(state_fns)
    run_dir = f'odom/{run_hash}/{run_date}/'
    save_state('odom/', state_fns)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    bdir = '/home/ronnypetson/Downloads/kitti_seq/dataset/'
    h, w = 376, 1241
    kp = KpT0_BA(h, w, bdir, seq_id)
    c = kp.camera_matrix
    c_ = np.linalg.inv(c)
    failure_eps = 1e-7
    poses = []
    poses_gt = []
    poses_ = []
    i = 0
    show_cloud = False
    pose0 = np.eye(4)
    W_poses = []
    cloud_all = np.zeros((3, 1))
    gT = np.zeros((kp.seq_len+1, 6))
    ge = np.zeros((kp.seq_len+1, 2))

    try:
        for i in range(kp.seq_len):
            i_ = min(i+1, kp.seq_len-1)
            kp.init_frame(i)
            T = kp._T0[i]
            Tgt = kp._Tgt[i]
            g = ba_graph(i, i+1)
            p = {}
            f = {}
            for ij in g:
                kp.init_BA(ij[0], ij[1])
                p[ij] = kp._kpts[ij[0]]
                f[ij] = kp._flow[ij]

            normT = np.linalg.norm(Tgt[:3, 3])

            poses.append(norm_t(T.copy(), normT))
            poses_gt.append(Tgt)

            x = {}
            x_ = {}
            for ij in g:
                x[ij] = p[ij][kp._vids[ij[0]], 0, :].transpose(1, 0) #
                z = np.ones((1, x[ij].shape[-1]))
                x[ij] = np.concatenate([x[ij], z], axis=0)

                x_[ij] = (p[ij] + f[ij])[kp._vids[ij[0]], 0, :].transpose(1, 0) #
                x_[ij] = np.concatenate([x_[ij], z], axis=0)

            opt = OptSingle(x, x_, c, g) # kp.E
            T0 = SE3.from_matrix(T, normalize=True)
            #T0 = T0.inv().log()
            T0 = T0.log()
            if i > 0:
                #gT[i] = (SE3.exp(gT[i-1]).dot(SE3.exp(T0))).log()
                gT[i] = (SE3.exp(T0).dot(SE3.exp(gT[i]))).log() # i -> i-1
            else:
                gT[i+1] = T0
            #T0 = np.zeros(6)
            foe0 = kp._ep0[i] / 1e3
            ge[i+1] = foe0
            #foe0 = np.array([607.1928, 185.2157]) / 1e3
            Tfoe = opt.optimize(gT, ge, freeze=True)
            #Tfoe = np.zeros((ge.shape[0], 8))

            print(opt.min_obj)
            if opt.min_obj > failure_eps and False:
                print('Initialization failure.')
                x = p[kp._avids[i], 0, :].transpose(1, 0)  # kp.avids
                z = np.ones((1, x.shape[-1]))
                x = np.concatenate([x, z], axis=0)

                x_ = p + f
                x_ = x_[kp._avids[i], 0, :].transpose(1, 0)  # kp.avids
                x_ = np.concatenate([x_, z], axis=0)

                opt = OptSingle(x, x_, c, None)
                T0 = np.zeros(6)
                #foe0 = np.array([w/2.0, h/2.0])
                foe0 = np.array([607.1928, 185.2157]) / 1e3
                Tfoe = opt.optimize(T0, foe0, freeze=False)
                print(f'New x0 status: {opt.min_obj <= failure_eps}')

            Tfoe = Tfoe.reshape(-1, 8)
            T_ = Tfoe[i+1, :6]
            foe = Tfoe[i+1, 6:]
            gT[i+1] = T_.copy()

            print(np.linalg.norm(gT[:i + 4]-Tfoe[:i + 4, :6]))
            input()

            #T_ = gT[i]
            print(foe)
            T_ = SE3.exp(T_).as_matrix() # .inv()
            if i > 0:
                T_ = T_ @ np.linalg.inv(SE3.exp(gT[i]).as_matrix())
            #T_ = norm_t(T_, normT)
            poses_.append(norm_t(T_.copy(), normT))
            pose0 = pose0 @ T_
            W_poses.append(pose0)

            if i % 10 == 9:
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

            i += 1
        save_poses(W_poses, f'{run_dir}/KITTI_{seq_id}.txt')
    except KeyboardInterrupt as e:
        save_poses(W_poses, f'{run_dir}/KITTI_{seq_id}.txt')
        raise e


if __name__ == '__main__':
    main()
