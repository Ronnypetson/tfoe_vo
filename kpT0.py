import numpy as np
import cv2
from liegroups import SE3
from glob import glob
from matplotlib import pyplot as plt
from opt import OptSingle
import pykitti

def homSE3tose3(R,t):
    ''' R is 3 x 3, t is 3 x 1
        se3 is 6 (t_3|r_3)
    '''
    p = np.zeros((4,4))
    p[:3,:3] = R
    p[:3,3:] = t
    p[3,3] = 1.0
    p = SE3.from_matrix(p,normalize=True)
    p = p.inv().log() #.inv()
    return p

def norm_t(T,norm):
    T[:3,3] /= np.linalg.norm(T[:3,3]) + 1e-8
    T[:3,3] *= norm
    return T

def T2traj(poses):
    p = np.eye(4)
    pts = [p[:,-1]]
    for T in poses:
        #T_ = SE3.exp(T) #.inv()
        #T_ = T_.as_matrix()
        #p = T_ @ p
        p = p @ T
        #p = np.linalg.inv(p) @ T_
        pts.append(p[:,-1])
    pts = np.array(pts)
    #pts = pts[:,:,0]
    return pts

def plot_traj(poses,poses_,outfn):
    pts = T2traj(poses)
    pts_ = T2traj(poses_)
    
    fig = plt.figure()
    plt.axis('equal')
    
    ax = fig.add_subplot(111)
    ax.plot(pts[:,0],pts[:,2],'g-')
    ax.plot(pts_[:,0],pts_[:,2],'b-')
    
    plt.savefig(outfn)
    plt.close(fig)

def plot_trajs(P,outfn,colors='gbr',glb=False):
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
    
    fig = plt.figure()
    plt.axis('equal')
    
    ax = fig.add_subplot(111)
    for i,p in enumerate(pts):
        ax.plot(p[:,0],p[:,2],f'{colors[i]}.')
    
    plt.savefig(outfn)
    plt.close(fig)

class KpT0:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self,h,w,basedir,seq):
        super().__init__()
        self.size = (h,w)
        self.kitti = pykitti.odometry(basedir,seq)
        self.gt_odom = self.kitti.poses
        
        #fig = plt.figure()
        #plt.axis('equal')
        #pts = np.array(self.gt_odom)[:,:3,3]
        #ax = fig.add_subplot(111)
        #ax.plot(pts[:,0],pts[:,2],'g-')
        #plt.savefig('testgt.png')
        #plt.close(fig)
        #input()
        
        #self.fns = sorted(glob(im_re),reverse=True)
        self.camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                      [0.0, 718.8560, 185.2157],
                                      [0.0, 0.0, 1.0]])
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=60,
                                                      nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                         criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 30, 0.03))
    
    def __iter__(self):
        camera_matrix = self.camera_matrix
        feature_detector = self.feature_detector
        lk_params = self.lk_params
        h,w = self.size
        gt_odom = self.gt_odom
        #fns = self.fns
        
        pts = []
        
        prev_image = np.array(next(self.kitti.cam0))
        prev_id = 0
        for i,image in enumerate(self.kitti.cam0):
            image = np.array(image)
            
            #image = cv2.imread(fns[i],0)
            #prev_image = cv2.imread(fns[i-1],0)
            
            prev_keypoint = feature_detector.detect(prev_image, None)
            points = np.array([[x.pt] for x in prev_keypoint],dtype=np.float32)
            
            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,image,points,\
                                                       None, **lk_params)
                
                E, mask = cv2.findEssentialMat(p1, points, camera_matrix,\
                                               cv2.RANSAC, 0.999, 0.1, None)
                
                self.vids = [j for j in range(len(mask)) if mask[j] == 1.0]
                
                _, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix, mask=mask) # , mask=mask
                T0 = np.eye(4)
                T0[:3,:3] = R
                T0[:3,3:] = t
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
    #seq_id = '2011_09_26_drive_0046_sync'
    #fn_re = f'/home/ronnypetson/Downloads/2011_09_26/{seq_id}/image_00/data/*.png'
    #fn_re = f'/home/ronnypetson/Downloads/kitti/image_0/*.png'
    seq_id = '01'
    bdir = '/home/ronnypetson/Downloads/kitti_seq/dataset/'
    kp = KpT0(376,1241,bdir,seq_id)
    c = kp.camera_matrix
    poses = []
    poses_gt = []
    poses_ = []
    i = 0
    for p,f,T,Tgt in kp:
        normT = np.linalg.norm(Tgt[:3,3])
        T = norm_t(T,normT)
        
        poses.append(T)
        poses_gt.append(Tgt)
        
        x = p[kp.vids,0,:].transpose(1,0) # 
        z = np.ones((1,x.shape[-1]))
        x = np.concatenate([x,z],axis=0)
        
        x_ = p + f
        x_ = x_[kp.vids,0,:].transpose(1,0) # kp.vids
        x_ = np.concatenate([x_,z],axis=0)
        
        opt = OptSingle(x,x_,c)
        #T0 = SE3.from_matrix(T).log()
        T0 = np.zeros(6)
        foe0 = np.array([1241/2,376/2])
        Tfoe = opt.optimize(T0,foe0)
        T_ = Tfoe[:6]
        T_ = SE3.exp(T_).inv().as_matrix()
        T_ = norm_t(T_,normT)
        poses_.append(T_)
        
        P = [poses_gt,poses,poses_]
        plot_trajs(P,f'{seq_id}.png',glb=False)

