import numpy as np
import cv2
from liegroups import SE3
from glob import glob
from matplotlib import pyplot as plt
from opt import OptSingle

def homSE3tose3(R,t):
    ''' R is 3 x 3, t is 3 x 1
        se3 is 6 (t_3|r_3)
    '''
    p = np.zeros((4,4))
    p[:3,:3] = R
    p[:3,3:] = t
    p[3,3] = 1.0
    p = SE3.from_matrix(p,normalize=True)
    p = p.inv().log()
    return p

def plot_traj(poses,outfn):
    p = np.zeros((4,1))
    p[-1,0] = 1.0
    pts = [p]
    for T in poses:
        T_ = SE3.exp(T)
        T_ = T_.as_matrix()
        p = T_ @ p
        pts.append(p)
    pts = np.array(pts)
    pts = pts[:,:,0]
    
    fig = plt.figure()
    plt.axis('equal')
    
    ax = fig.add_subplot(111)
    ax.plot(pts[:,0],pts[:,2],'b-')
    
    plt.savefig(outfn)
    plt.close(fig)

class KpT0:
    ''' Iterator class for returning key-points and pose initialization
    '''
    def __init__(self,h,w,im_re):
        super().__init__()
        self.size = (h,w)
        self.fns = sorted(glob(im_re),reverse=True)
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
        fns = self.fns
        
        for i in range(1,len(fns)):
            image = cv2.imread(fns[i],0)
            prev_image = cv2.imread(fns[i-1],0)
            #image = cv2.resize(image,(w,h))
            #prev_image = cv2.resize(prev_image,(w,h))
            
            #keypoint = feature_detector.detect(image, None)
            prev_keypoint = feature_detector.detect(prev_image, None)
            
            points = np.array([[x.pt] for x in prev_keypoint],dtype=np.float32)
            
            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,image,points,\
                                                       None, **lk_params)
                
                E, mask = cv2.findEssentialMat(p1, points, camera_matrix,\
                                               cv2.RANSAC, 0.999, 0.1, None)
                
                _, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix, mask=mask) # , mask=mask
                T0 = homSE3tose3(R,t)
                self.vids = [i for i in range(len(mask)) if mask[i] == 1.0]
            
            except cv2.error as e:
                print(e)
                yield None, None, None
            
            yield points, p1-points, T0

if __name__ == '__main__':
    fn_re = '/home/ronnypetson/Downloads/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/*.png'
    kp = KpT0(376,1241,fn_re)
    c = kp.camera_matrix
    poses = []
    poses_ = []
    i = 0
    for p,f,T in kp:
        poses.append(T)
        
        x = p[kp.vids,0,:].transpose(1,0)
        z = np.ones((1,x.shape[-1]))
        x = np.concatenate([x,z],axis=0)
        
        x_ = p + f
        x_ = x_[kp.vids,0,:].transpose(1,0)
        x_ = np.concatenate([x_,z],axis=0)
        
        opt = OptSingle(x,x_,c)
        T0 = np.zeros(6)
        foe0 = np.zeros(2)
        Tfoe = opt.optimize(T0,foe0)
        T_ = Tfoe[:6]
        T_[:3] /= np.linalg.norm(T_[:3])
        #T_ = SE3.exp(T_).as_matrix()
        poses_.append(T_)
        
        i += 1
        if i == 100:
            break
    plot_traj(poses,'traj.png')
    plot_traj(poses_,'traj_.png')

