import os
import sys
import time
import numpy as np
import cv2
import torch
from liegroups import SE3

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from opt_slocal import OptSingle
import pykitti
from versions import files_to_hash, save_state
from utils import norm_t, plot_trajs, ba_graph
from utils import save_poses, pt_cloud, plot_pt_cloud
from reproj import triangulate, rel_scale, rel_scale_
from reproj import triangulate_


class KpT0_BA:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self, h, w, basedir, seq, fast_th=25):
        super().__init__()
        self.size = (h, w)
        self.kitti = pykitti.odometry(basedir, seq)
        self.gt_odom = self.kitti.poses
        self.seq_len = len(self.gt_odom)
        #self.camera_matrix = np.array([[718.8560, 0.0, 607.1928],
        #                               [0.0, 718.8560, 185.2157],
        #                               [0.0, 0.0,      1.0]])
        self.camera_matrix = self.kitti.calib.K_cam0
        self.c = self.camera_matrix
        self.c2 = self.kitti.calib.K_cam1
        T_c10 = np.linalg.inv(self.kitti.calib.T_cam0_velo) @\
                     self.kitti.calib.T_cam1_velo
        T_vw = np.zeros((3, 3))
        T_vw[0, 1] = 1.0
        T_vw[1, 2] = 1.0
        T_vw[2, 0] = 1.0
        T_c10[:3, 3:] = T_vw @ T_c10[:3, 3:]
        self.T_c10 = T_c10
        ret = cv2.stereoRectify(self.c, np.zeros(4), self.c2,
                                np.zeros(4), self.size, self.T_c10[:3, :3],
                                self.T_c10[:3, 3:])
        self.Q = ret[4]

        self.c_ = np.linalg.inv(self.camera_matrix)
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=fast_th,
                                                               nonmaxSuppression=True)
        #self.lk_params = dict(winSize=(21, 21),
        #                      criteria=(cv2.TERM_CRITERIA_EPS |
        #                           cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 55, 0.03))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #self.stereo = cv2.StereoBM_create(numDisparities=1024, blockSize=15)
        window_size = 3
        min_disp = 16
        num_disp = 256 - min_disp
        self.min_disp = min_disp
        self.num_disp = num_disp
        self.stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                            numDisparities=num_disp,
                                            blockSize=16,
                                            P1=8 * 3 * window_size ** 2,
                                            P2=32 * 3 * window_size ** 2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32)

        self._kpts = {} # keypoint cache, i -> kp
        self._kptsij = {} # [i,j] -> kpij
        self._flow = {} # [i,j] -> f
        self._T0 = {} # i -> T
        self._ep0 = {} # i -> ep
        self._rs0 = {} # i -> rs
        self._trpt = {} # i -> [(p00, p10), (p01, p11)]
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
            im0r = self.kitti.get_cam1(i)
            im0r = np.array(im0r)

            im1 = self.kitti.get_cam0(i_)
            im1 = np.array(im1)
            im1r = self.kitti.get_cam1(i_)
            im1r = np.array(im1r)

            kp0 = self.feature_detector.detect(im0, None)
            kp0 = np.array([[x.pt] for x in kp0], dtype=np.float32)

            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(im0, im1, kp0,
                                                       None, **self.lk_params)
                #kp0r, _, _ = cv2.calcOpticalFlowPyrLK(im0, im0r, kp0,
                #                                       None, **self.lk_params)

                disp = self.stereo.compute(im0, im0r)

                Q = self.Q
                points = cv2.reprojectImageTo3D(disp, Q)
                #mask = disp > disp.min()
                #out_points = points[mask]
                #plt.imshow((disp - self.min_disp) / self.num_disp)
                #plt.show()

                kp0r = []
                sx = []
                for j in range(len(kp0)):
                    pt0r = np.array(kp0[j], dtype=np.int32)
                    dispj = disp[pt0r[0, 1], pt0r[0, 0]]
                    pt0r = np.array(pt0r, dtype=np.float32)
                    pt0r[0, 0] = pt0r[0, 0] - dispj
                    kp0r.append(pt0r) # 1.0 / np.abs(dispj)
                    if dispj > disp.min():
                        px = int(kp0[j, 0, 0])
                        py = int(kp0[j, 0, 1])
                        sx.append([points[py, px, 0],
                                   points[py, px, 1],
                                   points[py, px, 2]])
                kp0r = np.array(kp0r)
                sx = np.array(sx).T
                #sx = out_points.copy().T

                E, mask = cv2.findEssentialMat(p1, kp0, self.camera_matrix,
                                               cv2.RANSAC, 0.999, 0.1, None)
                vids = [j for j in range(len(mask)) if mask[j] == 1.0]
                avids = vids

                _, R, t, mask = cv2.recoverPose(E, p1, kp0, self.camera_matrix, mask=mask)
                T0 = np.eye(4)

                if np.trace(R) < 3*(1.0 - 0.2)\
                    or np.any(np.isnan(R))\
                    or np.any(np.isinf(R)):
                    R = np.eye(3)
                    t = np.zeros((3, 1))
                    t[-1] = 1.0

                if np.max(np.abs(t)) < 0.8\
                    or np.any(np.isnan(t))\
                    or np.any(np.isinf(t)):
                    t = np.zeros((3, 1))
                    t[-1] = 1.0

                T0[:3, :3] = R
                T0[:3, 3:] = t

                if len(mask) > 3:
                    vids_ = [j for j in range(len(mask)) if mask[j] == 1.0]
                    if len(vids_) > 3:
                        vids = vids_
                if np.median(np.abs(p1[vids] - kp0[vids])) < 1e-2 and False:
                    T0[:3, :3] = np.eye(3)
                    ep0 = self.camera_matrix[:3, 2:]
                elif np.sum(mask) < 64 and False: ###
                    T0[:3, :3] = np.eye(3)
                    ep0 = self.camera_matrix[:3, 2:]
                else:
                    ep2 = T0[:3, 3:]
                    ep0 = self.camera_matrix @ ep2
                ep0 = ep0[:, 0]

                inert = np.linalg.inv(self.gt_odom[i])
                Tgt = inert @ self.gt_odom[i_]

                if True:
                    #w, h = self.camera_matrix[:2, 2]
                    #spt = [j for j, p in enumerate(kp0)
                    #       if p[0, 1] > 3 * h // 2
                    #       and np.abs(p[0, 0] - w) < 230
                    #       and j in avids]
                    #if len(spt) < 8:
                    #    spt = [j for j, p in enumerate(kp0)
                    #           if p[0, 1] > 3 * h // 2
                    #           and np.abs(p[0, 0] - w) < 230]
                    ## and np.abs(p[0, 0] - w) < 300
                    #spt0 = kp0[avids][:, 0].T # spt
                    #spt1 = kp0r[avids][:, 0].T #p1[:][:, 0].T

                    #Ts = np.linalg.inv(T0.copy())
                    #Ts[:3, 3] /= np.linalg.norm(Ts[:3, 3])
                    #Ts = np.eye(4)
                    #Ts[0, -1] = 1.0
                    #Ts = np.linalg.inv(Ts)
                    #sx = triangulate_(spt0, spt1, Ts,
                    #                  self.camera_matrix)
                    #sx, den = triangulate(spt0, spt1, Ts,
                    #                      self.camera_matrix,
                    #                      self.camera_matrix @ Ts[:3, 3:])
                    #good = [j for j in range(sx.shape[1])
                    #        if sx[2, j] < 400
                    #        and sx[2, j] > 0.0]
                    # and den[j] > 0.05
                    #sx = sx[:, good]
                    #spt0 = spt0[:, good]

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.view_init(elev=90.0, azim=-90.0)
                    ax.scatter(sx[0], sx[1], sx[2], marker='.')
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    plt.show()

                    #c_ = np.linalg.inv(self.camera_matrix)
                    #spt0 = c_[:2, :2] @ spt0 + c_[:2, 2:]
                    #sc = 1.0 / np.abs(sx[2] * (spt0[1]))
                    #sc = np.median(sc) # / np.min(sc)
                    #sc2 = np.std(np.abs(sx[2]))
                    #if np.isnan(sc):
                    #    if i == 0:
                    #        self._rs0[i] = 1.0
                    #    else:
                    #        self._rs0[i] = self._rs0[i - 1]
                    #else:
                    #    self._rs0[i] = sc
                    #    #if i == 0:
                    #    #    self._rs0[i] = sc
                    #    #elif 0.9 < sc / self._rs0[i - 1] < 1.1:
                    #    #    self._rs0[i] = sc
                    #    #else:
                    #    #    self._rs0[i] = self._rs0[i - 1]

                    #sgt = np.linalg.norm(Tgt[:3, 3:])
                self._rs0[i] = 1.0

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
                self._moving[i] = True
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
    state_fns = ['kitti_slocal.py', 'opt_slocal.py', 'utils.py', 'reproj.py']
    run_hash = files_to_hash(state_fns)
    run_dir = f'odom/{run_hash}/{run_date}/'
    save_state('odom/', state_fns)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    bdir = '/home/ronnypetson/dataset/' # Downloads/kitti_seq/
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
    gT = np.zeros((kp.seq_len + 1, 6))
    ge = np.zeros((kp.seq_len + 1, 3))
    gs = np.ones((kp.seq_len + 1, 1))
    ge[0] = c @ np.array([0.0, 0.0, 1.0]) # / 1e3
    baw = 2
    kp.init_frame(0)
    sgt0 = np.linalg.norm(kp._Tgt[0][:3, 3])\
            / np.linalg.norm(kp._T0[0][:3, 3]) #kp._rs0[0]

    try:
        for i in range(0, kp.seq_len - (baw - 1), baw - 1):
            kp.init_frame(i)
            #Tgt = kp._Tgt[i]
            #T = kp._T0[i]

            g = ba_graph(i, i + (baw - 1))
            p = {}
            f = {}
            for ij in g:
                kp.init_BA(ij[0], ij[1])
                p[ij] = kp._kpts[ij[0]].copy()
                f[ij] = kp._flow[ij].copy()

            normT0 = np.linalg.norm(kp._Tgt[i][:3, 3].copy())
            #normT0 = kp._Tgt[i][2, 3]

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
                T0 = SE3.from_matrix(kp._T0[i + j].copy(), normalize=True).log()
                #T0[:3] = np.array([0.0, 0.0, 1.0])
                #T0 = np.zeros(6)
                norm_gt = np.linalg.norm(kp._Tgt[i + j][:3, 3].copy())
                #norm_gt = kp._Tgt[i + j][2, 3]
                sc = norm_gt / normT0
                scale_gt.append(sc)
                gT[i + j] = T0.copy()

            for j in range(baw):
                ge[i + j] = kp._ep0[i + j].copy() # / 1e3

            gs[i] = 1.0
            for j in range(i + 1, i + baw - 1, 1):
                continue
                #gs[j] = kp._rs0[j]
                #rs_ *= gs[j]
                #continue
                #id0 = j - 1
                #id1 = j
                #id2 = j + 1

                T01 = kp._Tgt[id0].copy() #
                T12 = kp._Tgt[id1].copy() #
                T02 = T12 @ T01 #

                #T01 = kp._T0[id0].copy() #
                #T12 = kp._T0[id1].copy() #
                #T02 = kp._Tij0[(id0, id2)].copy() #

                T01[:3, 3:] /= np.linalg.norm(T01[:3, 3])
                T12[:3, 3:] /= np.linalg.norm(T12[:3, 3])
                T02[:3, 3:] /= np.linalg.norm(T02[:3, 3])
                sc = rel_scale_(T01, T12, T02)
                rs_ *= sc
                rec_sc.append(rs_)

            Tfoe = opt.optimize(gT[i:i + baw],
                                ge[i:i + baw],
                                gs[i:i + baw],
                                freeze=True)
            print('loss', opt.min_obj)

            Tfoe = Tfoe.reshape(-1, 10)
            T_ = Tfoe[0, :6]
            foe = Tfoe[0, 6:9]
            sc = Tfoe[:, 9]

            #gT[i] = T_.copy()
            gT[i:i + baw] = Tfoe[:, :6]
            ge[i:i + baw] = Tfoe[:, 6:9]
            #gs[i:i + baw] = Tfoe[:, 9:]
            #gT[0] = Tfoe[0, :6]
            #ge[0] = Tfoe[0, 6:9]

            #s_ = Tfoe[:, 8]
            print('ep', Tfoe[:, 8])
            #print('scale\t', sc[:-1])
            #print('scalegt\t', scale_gt[:-1])

            for j in range(baw - 1):
                #normT = np.linalg.norm(kp._Tgt[i + j][:3, 3])
                #normT = rs0 * gs[i + j]
                normT = sgt0 * kp._rs0[i + j] #1.0
                poses.append(norm_t(kp._T0[i + j].copy(), normT))
                #poses_gt.append(norm_t(kp._Tgt[i + j].copy(), normT))
                poses_gt.append(kp._Tgt[i + j].copy())

            for j in range(baw - 1):
                #T_ = SE3.exp(T_).as_matrix() # .inv()
                T_ = SE3.exp(Tfoe[j, :6]).as_matrix()
                #normT = np.linalg.norm(kp._Tgt[i + j][:3, 3])
                normT = sgt0 * kp._rs0[i + j] #1.0
                #normT = rs0 * gs[i + j]
                poses_.append(norm_t(T_.copy(), normT))
                #poses_.append(T_.copy())
                #pose0 = pose0 @ T_
                pose0 = pose0 @ norm_t(T_.copy(), normT)
                #pose0_gt = pose0_gt @ norm_t(kp._Tgt[i + j].copy(), normT)
                pose0_gt = pose0_gt @ kp._Tgt[i + j].copy()
                pose0_init = pose0_init @ norm_t(kp._T0[i + j].copy(), normT)
                W_poses.append(pose0)
                W_poses_gt.append(pose0_gt)
                W_poses_init.append(pose0_init)

            if (i % baw == baw - 1) and (i % 10 == 9):
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
