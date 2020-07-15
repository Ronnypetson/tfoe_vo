import os
import sys
import time
import numpy as np
import cv2
import torch
from liegroups import SE3
from matplotlib import pyplot as plt
from opt import OptSingle
import pykitti
from geom import intersecc, null
from versions import files_to_hash, save_state
from utils import norm_t, plot_trajs
from utils import save_poses, pt_cloud, plot_pt_cloud


class KpT0:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self, h, w, basedir, seq):
        super().__init__()
        self.size = (h, w)
        self.kitti = pykitti.odometry(basedir, seq)
        self.gt_odom = self.kitti.poses
        self.camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                       [0.0, 718.8560, 185.2157],
                                       [0.0, 0.0,      1.0]])
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                               nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        #self.feature_detector = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def __iter__(self):
        camera_matrix = self.camera_matrix
        k_ = np.linalg.inv(camera_matrix)
        feature_detector = self.feature_detector
        lk_params = self.lk_params
        h, w = self.size
        gt_odom = self.gt_odom
        bf = self.bf

        pts = []

        prev_image = np.array(next(self.kitti.cam0))
        prev_image = cv2.resize(prev_image, (w, h))
        prev_id = 0
        for i, image in enumerate(self.kitti.cam0):
            image = np.array(image)
            image = cv2.resize(image, (w, h))

            prev_keypoint = feature_detector.detect(prev_image, None)
            points = np.array([[x.pt] for x in prev_keypoint], dtype=np.float32)

            #prev_keypoint, des1 = feature_detector.detectAndCompute(prev_image, None)
            #curr_keypoint, des2 = feature_detector.detectAndCompute(image, None)
            #matches = bf.match(des1, des2)
            #matches = sorted(matches, key=lambda x: x.distance)
            #points = np.array([[x.pt] for x in prev_keypoint], dtype=np.float32)

            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image, image, points,
                                                       None, **lk_params)

                #kp1 = []
                #kp2 = []
                #for j, m in enumerate(matches):
                #    t = m.trainIdx
                #    q = m.queryIdx
                #    pt1 = [prev_keypoint[q].pt[0], prev_keypoint[q].pt[1]]
                #    pt1 = np.array(pt1)
                #    pt2 = [curr_keypoint[t].pt[0], curr_keypoint[t].pt[1]]
                #    pt2 = np.array(pt2)
                #    kp1.append(pt1)
                #    kp2.append(pt2)
                #p1 = np.array(kp2)
                #p1 = np.expand_dims(p1, axis=1)
                #points = np.array(kp1)
                #points = np.expand_dims(points, axis=1)

                E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                               cv2.RANSAC, 0.999, 0.1, None)
                self.E = E

                self.vids = [j for j in range(len(mask)) if mask[j] == 1.0]
                self.avids = self.vids

                _, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix, mask=mask) # , mask=mask
                T0 = np.eye(4)
                T0[:3, :3] = R
                T0[:3, 3:] = t

                if len(mask) > 8:
                    vids_ = [j for j in range(len(mask)) if mask[j] == 1.0]
                    if len(vids_) > 8:
                        self.vids = vids_
                        self.avids = self.vids

                if i == 0:
                    T0 = np.eye(4)
                if np.mean(np.abs(p1[self.vids]-points[self.vids])) < 1e-2:
                    T0 = np.eye(4)
                    self.moving = False
                else:
                    self.moving = True
                #ep2 = np.linalg.inv(T0)[:3, 3:]
                ep2 = T0[:3, 3:]
                ep2 = ep2 / (ep2[-1] + 1e-8)
                ep2 = camera_matrix @ ep2
                self.ep0 = ep2[:2, 0]

                inert = np.linalg.inv(gt_odom[prev_id])
                Tgt = inert @ gt_odom[i]

                prev_image = image
                prev_id = i
                
            except cv2.error as e:
                print(e)
                yield None, None, None
            
            yield points, p1-points, T0, Tgt


def main():
    seq_id = sys.argv[1]
    run_date = time.asctime().replace(' ', '_')
    state_fns = ['kpT0.py', 'opt.py', 'utils.py', 'reproj.py', 'opt.py']
    run_hash = files_to_hash(state_fns)
    run_dir = f'odom/{run_hash}/{run_date}/'
    save_state('odom/', state_fns)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    bdir = '/home/ronnypetson/Downloads/kitti_seq/dataset/'
    h, w = 376, 1241
    kp = KpT0(h, w, bdir, seq_id)
    c = kp.camera_matrix
    c_ = np.linalg.inv(c)
    failure_eps = 1e-2
    poses = []
    poses_gt = []
    poses_ = []
    i = 0
    show_cloud = False
    pose0 = np.eye(4)
    W_poses = []
    cloud_all = np.zeros((3, 1))

    try:
        for p, f, T, Tgt in kp:
            normT = np.linalg.norm(Tgt[:3, 3])
            T = norm_t(T, normT)

            poses.append(T)
            poses_gt.append(Tgt)

            x = p[kp.vids, 0, :].transpose(1, 0) #
            z = np.ones((1, x.shape[-1]))
            x = np.concatenate([x, z], axis=0)

            x_ = p + f
            x_ = x_[kp.vids, 0, :].transpose(1, 0) # kp.vids
            x_ = np.concatenate([x_, z], axis=0)

            opt = OptSingle(x, x_, c, kp.E)
            T0 = SE3.from_matrix(T, normalize=True)
            T0 = T0.inv().log()
            #T0 = np.zeros(6)
            foe0 = kp.ep0 / 1e3
            #foe0 = np.array([607.1928, 185.2157]) / 1e3
            if kp.moving:
                Tfoe = opt.optimize(T0, foe0, freeze=False)
            else:
                Tfoe = np.zeros(8)
                Tfoe[6:] = foe0
                opt.min_obj = 0.0

            print(opt.min_obj)
            if opt.min_obj > failure_eps:
                print('Initialization failure.')
                x = p[kp.vids, 0, :].transpose(1, 0)  # kp.avids
                z = np.ones((1, x.shape[-1]))
                x = np.concatenate([x, z], axis=0)

                x_ = p + f
                x_ = x_[kp.vids, 0, :].transpose(1, 0)  # kp.avids
                x_ = np.concatenate([x_, z], axis=0)

                opt = OptSingle(x, x_, c, kp.E)
                T0 = np.zeros(6)
                #foe0 = np.array([w/2.0, h/2.0])
                foe0 = np.array([607.1928, 185.2157]) / 1e3
                Tfoe = opt.optimize(T0, foe0, freeze=False)
                print(f'New x0 status: {opt.min_obj <= failure_eps}')
                #Tfoe = np.zeros(8) ###

            T_ = Tfoe[:6]
            foe = Tfoe[6:]
            print(foe)
            T_ = SE3.exp(T_).inv().as_matrix()
            T_ = norm_t(T_, normT)
            poses_.append(T_)
            pose0 = pose0 @ T_
            W_poses.append(pose0)

            if i % 30 == 29:
                P = [poses_gt, poses, poses_]
                plot_trajs(P, f'{run_dir}/{seq_id}.svg', glb=False)

            if show_cloud:
                scale = normT / (np.linalg.norm(T_[:3, 3])+1e-8)
                p = torch.from_numpy(p[kp.vids, 0]).double()
                p_ = p + torch.from_numpy(f[kp.vids, 0]).double()
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
