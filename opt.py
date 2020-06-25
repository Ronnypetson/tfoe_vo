import numpy as np
from scipy.optimize import minimize
from reproj import reproj, depth, gen_pts, reproj_tc, depth_tc, reproj_tc_foe
from liegroups import SE3
import torch
from liegroups.torch import SE3 as SE3tc

class OptSingle:
    def __init__(self,x,x_,c):
        ''' x and x_ are 3xN
        '''
        self.x = x
        self.x_ = x_
        self.f = x_-x
        n = x.shape[1]
        self.T = np.zeros(6)
        self.foe = np.zeros(2)
        self.c = torch.from_numpy(c).float()
        self.c_ = torch.inverse(self.c)
    
    def objective(self,Tfoe,grad=False):
        ''' Tfoe is 8
        '''
        Tfoe = torch.from_numpy(Tfoe)
        Tfoe = Tfoe.float()
        Tfoe = Tfoe.requires_grad_(True)
        Tfoe.retain_grad()
        Tfoe_ = Tfoe.clone()
        
        T = Tfoe_[:6]
        foe = Tfoe_[6:]
        foe = foe.unsqueeze(-1)
        T = SE3tc.exp(T.clone())
        T = T.as_matrix()
        
        c = self.c
        c_ = self.c_
        
        #d = depth_tc(torch.from_numpy(self.x[:2]).float(),\
        #             torch.from_numpy(self.f[:2]).float(),foe)
        #x_rep = reproj_tc(torch.from_numpy(self.x).float(),T,d,c)
        #y = torch.from_numpy(self.x_).float()-x_rep
        
        x_rep = reproj_tc_foe(torch.from_numpy(self.x).float(),\
                              torch.from_numpy(self.x_).float(),\
                              T,foe,c)
        y = c_ @ torch.from_numpy(self.x_).float()-x_rep
        
        y = torch.mean(torch.abs(y))
        #y = y + torch.abs(1.0 - torch.norm(Tfoe_[:3]))
        y.backward()
        #if grad:
        gradTfoe = Tfoe.grad.detach().numpy()
        y = y.detach().numpy()
        #print(y)
        return y, gradTfoe
    
    def optimize(self,T0,foe0):
        ''' T0 is 6
            foe0 is 2
        '''
        Tfoe0 = np.concatenate([T0,foe0],axis=0)
        Tfoe0 = np.expand_dims(Tfoe0,axis=-1)
        res = minimize(self.objective,\
                       Tfoe0,method='BFGS',\
                       jac=True,\
                       options={'disp': True,\
                                'maxiter':100,\
                                'gtol':1e-8})
        return res.x

if __name__ == '__main__':
    x,x_,d,R,t = gen_pts(3)
    
    opt = OptSingle(x,x_)
    T0 = np.zeros(6)
    foe0 = np.zeros(2)
    with torch.autograd.set_detect_anomaly(False):
        Tfoe = opt.optimize(T0,foe0)
    T = Tfoe[:6]
    foe = Tfoe[6:]
    T = SE3.exp(T).as_matrix()
    print(T[:3,:3])
    print(R)
    print(foe)
    print()
    print(T[:3,3:])
    print(t)

