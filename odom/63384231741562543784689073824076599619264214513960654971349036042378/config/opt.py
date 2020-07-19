import numpy as np
from scipy.optimize import minimize
from reproj import reproj, depth, gen_pts, E_from_T
from reproj import reproj_tc, depth_tc, reproj_tc_foe
from reproj import reproj_tc_
from liegroups import SE3
import torch
import torch.nn.functional as F
from liegroups.torch import SE3 as SE3tc
from hessian import jacobian, hessian


class OptSingle:
    def __init__(self, x, x_, c, E):
        ''' x and x_ are 3xN
        '''
        self.x = x
        self.x_ = x_
        self.f = x_-x
        n = x.shape[1]
        self.T = np.zeros(6)
        self.foe = np.zeros(2)
        self.c = torch.from_numpy(c) #.float()
        self.c_ = torch.inverse(self.c)
        self.min_obj = np.inf
    
    def obj_tc(self, Tfoe):
        T = Tfoe[:6]
        foe = Tfoe[6:]
        foe = foe.unsqueeze(-1)
        T = SE3tc.exp(T.clone())
        T = T.as_matrix()
        c = self.c
        c_ = self.c_
        x_rep = reproj_tc_foe(torch.from_numpy(self.x),
                              torch.from_numpy(self.x_),
                              T, foe, c)
        y = c_ @ torch.from_numpy(self.x_)-x_rep
        y = torch.mean(y**2.0)
        return y
   
    def obj_npy(self, Tfoe):
        Tfoe = torch.from_numpy(Tfoe)
        Tfoe = Tfoe.requires_grad_(True)
        Tfoe.retain_grad()
        Tfoe_ = Tfoe.clone()
        y = self.obj_tc(Tfoe_)
        y = y.detach().numpy()
        return y
    
    def jac_npy(self, Tfoe):
        Tfoe = torch.from_numpy(Tfoe)
        Tfoe = Tfoe.requires_grad_(True)
        jac = jacobian(self.obj_tc(Tfoe),Tfoe)
        jac = jac.detach().numpy()
        return jac
    
    def hess_npy(self, Tfoe):
        Tfoe = torch.from_numpy(Tfoe)
        Tfoe = Tfoe.requires_grad_(True)
        hess = hessian(self.obj_tc(Tfoe), Tfoe)
        hess = hess.detach().numpy()
        return hess
    
    def objective(self, Tfoe, grad=False):
        ''' Tfoe is 8
        '''
        Tfoe = torch.from_numpy(Tfoe)
        #Tfoe = Tfoe.float()
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

        #x_rep = reproj_tc_(torch.from_numpy(self.x),
        #                   torch.from_numpy(self.x_),
        #                   T, foe, c)

        x_rep = reproj_tc_foe(torch.from_numpy(self.x),
                              torch.from_numpy(self.x_),
                              T, foe, c)

        #y = c_ @ torch.from_numpy(self.x_) - x_rep
        #y = torch.mean(y**2.0) # + epi_loss

        #t = torch.inverse(T)[:3, 3:]
        #t = t / (torch.norm(t) + 1e-8)
        #z = torch.ones(1, 1).double()
        #foe_ = foe * 100.0  ###
        #foe_ = torch.cat([foe_, z], dim=0)
        #foe_ = c_ @ foe_
        #foe_ = foe_ / torch.norm(foe_)
        #reg = 1.0 - (t.T @ foe_)[0, 0]

        y = F.smooth_l1_loss(c_ @ torch.from_numpy(self.x_), x_rep) # + 1e-1*reg
        y.backward()
        gradTfoe = Tfoe.grad.detach().numpy()
        y = y.detach().numpy()
        self.min_obj = min(self.min_obj, y)
        return y, gradTfoe

    def optimize(self, T0, foe0, freeze=True):
        ''' T0 is 6
            foe0 is 2
        '''
        self.min_obj = np.inf
        Tfoe0 = np.concatenate([T0, foe0], axis=0)
        Tfoe0 = np.expand_dims(Tfoe0, axis=-1)
        
        #res = minimize(self.objective,
        #               Tfoe0, method='BFGS',
        #               jac=True,
        #               options={'disp': False,
        #                        'maxiter': 1000,
        #                        'gtol': 1e-10})

        bounds = []
        if freeze:
            #for par in Tfoe0:
            #    bounds.append((par-1e-5, par+1e-5))
            for par in Tfoe0[:6]:
                bounds.append((par-1e-10, par+1e-10))
            for par in Tfoe0[6:]:
                bounds.append((None, None))
        else:
            for par in Tfoe0:
                bounds.append((None, None))

        res = minimize(self.objective,
                       Tfoe0, method='L-BFGS-B',
                       jac=True,
                       bounds=bounds,
                       #tol=1e-14,
                       options={'disp': False,
                                'maxiter': 1e3,
                                'gtol': 1e-10,
                                'ftol': 1e-10})

        #res = minimize(self.objective,
        #               Tfoe0, method='BFGS',
        #               jac=True,
        #               tol=1e-12,
        #               options={'disp': False,
        #                        'maxiter': 1e3,
        #                        'gtol': 1e-12})
        
        #res = minimize(self.obj_npy,
        #               Tfoe0, method='Newton-CG',
        #               jac=self.jac_npy,
        #               hess=self.hess_npy,
        #               options={'disp': False,
        #                        'maxiter': 1e3})

        return res.x


if __name__ == '__main__':
    x, x_, d, R, t = gen_pts(3)
    
    opt = OptSingle(x, x_)
    T0 = np.zeros(6)
    foe0 = np.zeros(2)
    with torch.autograd.set_detect_anomaly(False):
        Tfoe = opt.optimize(T0, foe0)
    T = Tfoe[:6]
    foe = Tfoe[6:]
    T = SE3.exp(T).as_matrix()
    print(T[:3, :3])
    print(R)
    print(foe)
    print()
    print(T[:3, 3:])
    print(t)
