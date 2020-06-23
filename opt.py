import numpy as np
from scipy.optimize import minimize
from reproj import reproj, depth, gen_pts
from liegroups import SE3

class OptSingle:
    def __init__(self,x,x_):
        ''' x and x_ are 3xN
        '''
        self.x = x
        self.x_ = x_
        self.f = x_-x
        n = x.shape[1]
        self.T = np.zeros(6)
        self.foe = np.zeros(2)
    
    def objective(self,Tfoe):
        ''' Tfoe is 8
        '''
        T = Tfoe[:6]
        foe = Tfoe[6:]
        foe = np.expand_dims(foe,axis=-1)
        T = SE3.exp(T).as_matrix()
        d = depth(self.x[:2],self.f[:2],foe)
        c = np.eye(3)
        x_rep = reproj(self.x,T,d,c)
        y = self.x_-x_rep
        y = np.linalg.norm(y)
        print(y)
        return y

    def obj_grad(self,poses_foes):
        pass
    
    def optimize(self,T0,foe0):
        ''' T0 is 6
            foe0 is 2
        '''
        Tfoe0 = np.concatenate([T0,foe0],axis=0)
        Tfoe0 = np.expand_dims(Tfoe0,axis=-1)
        res = minimize(self.objective,\
                       Tfoe0,method='Nelder-Mead',\
                       jac=None,\
                       options={'disp': True,\
                                'maxiter':1000})
        return res.x

if __name__ == '__main__':
    x,x_,d,R,t = gen_pts(5)
    opt = OptSingle(x,x_)
    T0 = np.zeros(6)
    foe0 = np.zeros(2)
    opt.optimize(T0,foe0)

