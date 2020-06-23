import numpy as np
import cv2
from liegroups import SE3

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

def depth(p,f,foe):
    ''' p is 2xN
        f is 2xN (flow)
        foe is 2x1
    '''
    mag = np.linalg.norm(f,axis=0)
    dist = p-foe
    dist = np.linalg.norm(dist,axis=0)
    d = dist / mag
    # d = np.linalg.norm(foe-(w/2,h/2))/mag
    return d

def gen_pts(n):
    ''' returns x,x'
        x and x' are 3xN
    '''
    x = np.random.randn(3,n)
    d = x[[-1]]
    x = x / d
    
    R = np.random.randn(3,3)
    U,S,Vt = np.linalg.svd(R)
    R = Vt.T
    t = np.random.randn(3,1)
    
    x_ = R @ (x*d) + t
    x_ = x_ / x_[[-1]]
    
    return x,x_,d,R,t

if __name__ == '__main__':
    x,x_,d,R,t = gen_pts(5)
    f = x_-x
    c = np.eye(3)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3:] = t
    x_rep = reproj(x,T,d,c)
    print(np.linalg.norm(x_-x_rep))
    foe = np.zeros((2,1))
    d_ = depth(x[:2],f,foe)
    x_rep = reproj(x,T,d_,c)
    print(np.linalg.norm(x_-x_rep))

