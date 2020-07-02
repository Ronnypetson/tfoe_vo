import numpy as np
import cv2
from liegroups import SO3
from matplotlib import pyplot as plt


def cam0(dir, stampsfn, c, dist, newc):
    with open(stampsfn, 'r') as f:
        lines = f.read().split('\n')
        for i, l in enumerate(lines[1:]):
            fn = l.split(',')[-1]
            imfn = f'{dir}/{fn}'
            im = cv2.imread(imfn, 0)
            im = cv2.undistort(im, c, dist, newCameraMatrix=newc)
            yield im


class EuRoC:
    def __init__(self, basedir, seq_id):
        dir = f'{basedir}/{seq_id}/mav0/cam0/data/'
        stampsfn = f'{basedir}/{seq_id}/mav0/cam0/data.csv'
        odomfn = f'{basedir}/{seq_id}/mav0/state_groundtruth_estimate0/data.csv'

        self.c0 = np.array([[458.654, 0.0, 367.215],
                            [0.0, 457.296, 248.375],
                            [0.0, 0.0,     1.0]])
        self.newc = self.c0.copy()
        #self.newc[0, 0] = self.c0[0, 0] / np.sqrt(2)
        #self.newc[1, 1] = self.c0[1, 1] / np.sqrt(2)
        # radial-tangential distortion
        self.dist = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

        self.TR = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                            [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                            [0.0, 0.0, 0.0, 1.0]])

        self.cam0 = cam0(dir, stampsfn, self.c0, self.dist, self.newc)
        self.poses = np.array(self._get_odom(odomfn))

    def _get_odom(self, odomfn):
        T_ = np.zeros((4, 4))
        T_[0, 1] = 1.0
        T_[1, 2] = 1.0
        T_[2, 0] = 1.0
        T_[3, 3] = 1.0
        self.T_ = T_

        odom = []
        with open(odomfn, 'r') as f:
            lines = f.read().split('\n')
            for i, l in enumerate(lines[1:]):
                T = l.split(',')[1:8]
                if len(T) == 7:
                    T = [float(t) for t in T]
                    T = np.array(T)
                    Tt = T[:3]
                    qnorm = np.linalg.norm(T[3:])
                    Tr = T[3:] / qnorm
                    Tr = SO3.from_quaternion(Tr, 'wxyz')
                    Tr = Tr.as_matrix()
                    T = np.eye(4)
                    T[:3, :3] = Tr
                    T[:3, 3] = Tt
                    T = (T @ self.TR) @ self.T_
                    odom.append(T)
        return odom

