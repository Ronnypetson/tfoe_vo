import numpy as np
import cv2
import torch
from liegroups import SE3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from opt import OptSingle
import pykitti
from reproj import depth_tc
from euroc.loader import EuRoC


def homSE3tose3(R, t):
    ''' R is 3 x 3, t is 3 x 1
        se3 is 6 (t_3|r_3)
    '''
    p = np.zeros((4, 4))
    p[:3, :3] = R
    p[:3, 3:] = t
    p[3, 3] = 1.0
    p = SE3.from_matrix(p, normalize=True)
    p = p.inv().log() #.inv()
    return p


def norm_t(T, norm):
    T[:3, 3] /= np.linalg.norm(T[:3, 3]) + 1e-8
    T[:3, 3] *= norm
    return T


def T2traj(poses):
    p = np.eye(4)
    pts = [p[:, -1]]
    for T in poses:
        p = p @ T
        pts.append(p[:, -1])
    pts = np.array(pts)
    return pts


def pt_cloud(p, p_, T, foe, scale, c):
    ''' p is 2,N
        p_ is 2,N
        T is 4,4
        foe is 2,1
        scale is a scalar
        c is 3,3
    '''
    p = p.permute(1, 0)
    p_ = p_.permute(1, 0)
    z = torch.ones(1, p.size(-1)).double()
    z_ = torch.ones(1, 1).double()
    p = torch.cat([p, z], dim=0)
    p_ = torch.cat([p_, z], dim=0)
    foe = torch.cat([foe, z_], dim=0)
    
    c_ = torch.inverse(c)
    p = c_ @ p
    p_ = c_ @ p_
    foe = c_ @ foe
    
    d = depth_tc(p[:2], (p_-p)[:2], foe[:2])
    d *= scale
    
    x = p * d
    x = T[:3, :3] @ x + T[:3, 3:]
    #thresh_d = 5*torch.min(d)
    #close = (d < thresh_d).nonzero()
    #close = close.reshape(-1)
    _, close = torch.topk(-d, k=10)
    
    x = x[:, close]
    x = x.detach().numpy()
    return x


def plot_pt_cloud(x, outfn):
    fig = plt.figure()
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Z Label')
    ax.set_ylabel('X Label')
    ax.set_zlabel('Y Label')
    ax.view_init(azim=177, elev=65)
    ax.scatter(x[2, :], -x[0, :], x[1, :], marker='.')
    ax_data = ax.plot([0], [0], [0], marker='.')[0]
    plt.show()
    
    #plt.savefig(outfn)
    #plt.close(fig)


def plot_traj(poses, poses_, outfn):
    pts = T2traj(poses)
    pts_ = T2traj(poses_)
    
    fig = plt.figure()
    plt.axis('equal')
    
    ax = fig.add_subplot(111)
    ax.plot(pts[:, 0], pts[:, 2], 'g-')
    ax.plot(pts_[:, 0], pts_[:, 2], 'b-')
    
    plt.savefig(outfn)
    plt.close(fig)


def plot_trajs(P, outfn, colors='gbr', glb=False):
    ''' P is k,n,6
    '''
    pts = []
    if not glb:
        for p in P:
            pts.append(T2traj(p))
    else:
        for p in P:
            pts.append(p)
    pts = np.array(pts)
    
    #fig = plt.figure()
    fig, axs = plt.subplots(3) #3
    plt.axis('equal')
    
    #ax = fig.add_subplot(111)
    #ax2 = fig.add_subplot(111)
    for i, p in enumerate(pts):
        axs[0].plot(p[:, 0], p[:, 2], f'{colors[i]}-')
        axs[1].plot(p[:, 0], p[:, 1], f'{colors[i]}-')
        axs[2].plot(p[:, 1], p[:, 2], f'{colors[i]}-')
    
    plt.savefig(outfn)
    plt.close(fig)


class KpT0:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self, h, w, bdir, seq_id):
        super().__init__()
        self.size = (h, w)
        self.euroc = EuRoC(bdir, seq_id) #pykitti.odometry(basedir, seq)
        self.gt_odom = self.euroc.poses
        self.Tc = self.euroc.T_
        self.camera_matrix = np.array([[458.654, 0.0, 367.215],
                                       [0.0, 457.296, 248.375],
                                       [0.0, 0.0,     1.0]])
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                               nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 30, 0.03))
    
    def __iter__(self):
        camera_matrix = self.camera_matrix
        feature_detector = self.feature_detector
        lk_params = self.lk_params
        h, w = self.size
        gt_odom = self.gt_odom
        #fns = self.fns
        
        pts = []

        prev_image = np.array(next(self.euroc.cam0))
        prev_image = cv2.resize(prev_image, (w, h))
        prev_id = 0
        for i, image in enumerate(self.euroc.cam0):
            print(i)
            image = np.array(image)

            prev_keypoint = feature_detector.detect(prev_image, None)
            points = np.array([[x.pt] for x in prev_keypoint], dtype=np.float32)

            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image, image, points,
                                                       None, **lk_params)

                E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                               cv2.RANSAC, 0.999, 0.1, None)
                
                self.vids = [j for j in range(len(mask)) if mask[j] == 1.0]
                self.avids = [j for j in range(len(st)) if st[j] == 1.0 and mask[j] == 0.0]
                if len(self.avids) < 3:
                    self.avids = [j for j in range(len(points)) if mask[j] == 0.0]
                if len(self.avids) < 3:
                    self.avids = [j for j in range(len(points))]

                _, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix, mask=mask) # , mask=mask
                T0 = np.eye(4)
                T0[:3, :3] = R
                T0[:3, 3:] = t
                #T0 = np.linalg.inv(T0)
                if i == 0:
                    T0 = np.eye(4)
                #T0 = homSE3tose3(T0[:3,:3],T0[:3,3:])
                #T0 = homSE3tose3(R,t)
                #T0 = SE3.exp(T0).as_matrix()
                
                inert = np.linalg.inv(gt_odom[prev_id])
                #Tgt = gt_odom[i] @ inert
                Tgt = inert @ gt_odom[i]
                #Tgt = gt_odom[prev_id] @ Tgt
                #Tgt = homSE3tose3(Tgt[:3,:3],Tgt[:3,3:])
                
                #normT = np.linalg.norm(Tgt[:3,3])
                #T0 = norm_t(T0,normT)
                #pts.append(Tgt)
                #plot_trajs([pts],'testgt_.png',glb=False)
                
                prev_image = image
                prev_id = i
                
            except cv2.error as e:
                print(e)
                yield None, None, None
            
            yield points, p1-points, T0, Tgt


if __name__ == '__main__':
    seq_id = 'V2_01_easy' #'MH_01_easy'
    bdir = '/home/ronnypetson/Downloads/'
    h, w = 480, 752
    kp = KpT0(h, w, bdir, seq_id)
    c = kp.camera_matrix
    failure_eps = 5e-1
    poses = []
    poses_gt = []
    poses_ = []
    i = 0
    pose0 = np.eye(4)
    cloud_all = np.zeros((3, 1))
    
    for p, f, T, Tgt in kp:
        #T = T @ kp.Tc
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
        
        opt = OptSingle(x, x_, c)
        #T0 = SE3.from_matrix(T).inv().log()
        T0 = np.zeros(6)
        foe0 = np.array([w/2.0, h/2.0])
        #foe0 = foe0 + 1e1*np.random.randn(2) ###
        Tfoe = opt.optimize(T0, foe0)

        if opt.min_obj > failure_eps:
            print('Initialization failure.')
            x = p[kp.avids, 0, :].transpose(1, 0)  #
            z = np.ones((1, x.shape[-1]))
            x = np.concatenate([x, z], axis=0)

            x_ = p + f
            x_ = x_[kp.avids, 0, :].transpose(1, 0)  # kp.vids
            x_ = np.concatenate([x_, z], axis=0)

            opt = OptSingle(x, x_, c)
            T0 = np.zeros(6)
            foe0 = np.array([w / 2.0, h / 2.0])
            # foe0 = foe0 + 1e1*np.random.randn(2) ###
            Tfoe = opt.optimize(T0, foe0, freeze=False)
            print(f'New x0 status: {opt.min_obj <= failure_eps}')

        T_ = Tfoe[:6]
        foe = Tfoe[6:]
        print(foe)
        T_ = SE3.exp(T_).inv().as_matrix()
        #T_ = T_ @ kp.Tc
        T_ = norm_t(T_, normT)
        poses_.append(T_)
        pose0 = pose0 @ T_
        
        if i % 30 == 29:
            P = [poses_gt, poses, poses_]
            plot_trajs(P, f'euroc_{seq_id}.png', glb=False)

        if False:
            scale = normT / (np.linalg.norm(T_[:3, 3])+1e-8)
            p = torch.from_numpy(p[kp.vids, 0]).double()
            p_ = p + torch.from_numpy(f[kp.vids, 0]).double()
            T_acc = torch.from_numpy(pose0).double()
            foe = torch.from_numpy(foe).double().unsqueeze(-1)
            c_tc = torch.from_numpy(c).double()
            cloud = pt_cloud(p, p_, T_acc, foe, scale, c_tc)
            cloud_all = np.concatenate([cloud_all, cloud], axis=1) # [:,:20]
        
        if i % 80 == 79 and False:
            plot_pt_cloud(np.array(cloud_all), f'euroc_{seq_id}_pt_cloud.svg')
        
        i += 1
