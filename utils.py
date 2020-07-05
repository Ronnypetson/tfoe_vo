import numpy as np
import torch
from matplotlib import pyplot as plt
from liegroups import SE3
from reproj import depth_tc


def homSE3tose3(R, t):
    ''' R is 3 x 3, t is 3 x 1
        se3 is 6 (t_3|r_3)
    '''
    p = np.zeros((4, 4))
    p[:3, :3] = R
    p[:3, 3:] = t
    p[3, 3] = 1.0
    p = SE3.from_matrix(p, normalize=True)
    p = p.inv().log()  # .inv()
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


def save_poses(p, outfn):
    ''' p is in homogeneous format
    '''
    with open(outfn, 'w') as f:
        for T in p:
            T_ = T.reshape(-1)[:12]
            T_ = T_.tolist()
            T_ = [str(t) for t in T_]
            T_ = ' '.join(T_) + '\n'
            f.write(T_)


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

    d = depth_tc(p[:2], (p_ - p)[:2], foe[:2])
    d *= scale

    x = p * d
    x = T[:3, :3] @ x + T[:3, 3:]
    # thresh_d = 5*torch.min(d)
    # close = (d < thresh_d).nonzero()
    # close = close.reshape(-1)
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

    # plt.savefig(outfn)
    # plt.close(fig)


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

    # fig = plt.figure()
    fig, axs = plt.subplots(1)  # 3
    plt.axis('equal')

    # ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(111)
    for i, p in enumerate(pts):
        axs.plot(p[:, 0], p[:, 2], f'{colors[i]}-')
        # axs[1].plot(p[:,0],p[:,1],f'{colors[i]}-')
        # axs[2].plot(p[:,1],p[:,2],f'{colors[i]}-')

    plt.savefig(outfn)
    plt.close(fig)
