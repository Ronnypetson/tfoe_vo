import numpy as np
import cv2
from liegroups import SE3
import torch

def reproj(p,T,d,c):
    ''' Computes rerpojection of points p given pose T and camera intrinsics c.
        T is in SE3 form.
        d is 1xN -- point depths.
        c is in 3x3 form.
        p is 3xN in image coordinates.
    '''
    c_ = np.linalg.inv(c)
    p = c_ @ p
    x = p * d
    x_ = T[:3,:3] @ x + T[:3,3:]
    p_ = x_ / x_[-1:]
    p_ = c @ p_
    return p_

def reproj_tc(p,T,d,c):
    ''' Computes rerpojection of points p given pose T and camera intrinsics c.
        T is in SE3 form.
        d is 1xN -- point depths.
        c is in 3x3 form.
        p is 3xN in image coordinates.
    '''
    c_ = torch.inverse(c)
    p = c_ @ p
    x = p * d
    x_ = T[:3,:3] @ x + T[:3,3:]
    p_ = x_ / x_[-1:]
    p_ = c @ p_
    return p_

def reproj_tc_foe(p,p_,T,foe,c):
    ''' Computes rerpojection of points p given pose T and camera intrinsics c.
        T is in SE3 form.
        d is 1xN -- point depths.
        c is in 3x3 form.
        p is 3xN in image coordinates.
    '''
    c_ = torch.inverse(c)
    p = c_ @ p
    
    p_ = c_ @ p_
    z = torch.ones(1,1).double()
    foe = torch.cat([foe,z],dim=0)
    foe = c_ @ foe
    d = depth_tc(p[:2],(p_-p)[:2],foe[:2])
    
    x = p * d
    x_ = T[:3,:3] @ x + T[:3,3:]
    p_rep = x_ / x_[-1:]
    #p_rep = c @ p_rep
    return p_rep

def depth(p,f,foe):
    ''' p is 2xN
        f is 2xN (flow)
        foe is 2x1
    '''
    mag = np.linalg.norm(f,axis=0)
    dist = p-foe
    dist = np.linalg.norm(dist,axis=0)
    d = dist / mag
    return d

def depth_tc(p,f,foe):
    ''' p is 2xN
        f is 2xN (flow)
        foe is 2x1
    '''
    mag = torch.norm(f,dim=0)
    mag = torch.clamp(mag,0.001)
    dist = p-foe
    dist = torch.norm(dist,dim=0)
    d = dist / mag
    return d

def gen_pts(n):
    ''' returns x,x'
        x and x' are 3xN
    '''
    x = 1e3*np.random.randn(3,n)
    x[-1] = 1.0
    d = 1e3*np.abs(np.random.randn(1,n))
    
    tse3 = 1e-2*np.random.randn(6)
    tse3[[3,5]] = 0.0
    T = SE3.exp(tse3)
    T = T.as_matrix()
    R = T[:3,:3]
    t = T[:3,3:]
    
    #print(x*d)
    #print(R)
    #print(R @ (x*d))
    #print()
    
    x_ = R @ (x*d) + t
    x_ = x_ / x_[[-1]]
    
    return x,x_,d,R,t

if __name__ == '__main__':
    x,x_,d,R,t = gen_pts(60)
    f = x_-x
    c = np.eye(3)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3:] = t
    
    x_rep = reproj_tc(torch.from_numpy(x),\
                      torch.from_numpy(T),\
                      torch.from_numpy(d),\
                      torch.from_numpy(c))
    print(np.linalg.norm(x_-x_rep.detach().numpy()))
    
    x_rep = reproj(x,T,d,c)
    print(np.linalg.norm(x_-x_rep))
    foe = np.zeros((2,1))
    d_ = depth(x[:2],f,foe)
    x_rep = reproj(x,T,d_,c)
    print(np.linalg.norm(x_-x_rep))

