import numpy as np
import cv2
from liegroups import SO3


def cam0(dir, stampsfn):
    with open(stampsfn, 'r') as f:
        lines = f.read().split('\n')
        for l in lines[1:]:
            fn = l.split(',')[-1]
            imfn = f'{dir}/{fn}'
            im = cv2.imread(imfn, 0)
            yield im


class EuRoC:
    def __init__(self, basedir, seq_id):
        # '/home/ronnypetson/Downloads/'
        dir = f'{basedir}/{seq_id}/mav0/cam0/data/'
        stampsfn = f'{basedir}/{seq_id}/mav0/cam0/data.csv'
        odomfn = f'{basedir}/{seq_id}/mav0/state_groundtruth_estimate0/data.csv'
        self.cam0 = cam0(dir, stampsfn)
        self.poses = np.array(self._get_odom(odomfn))

    def _get_odom(self, odomfn):
        T_ = np.eye(4)
        odom = [T_]
        with open(odomfn, 'r') as f:
            lines = f.read().split('\n')
            for l in lines[1:]:
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
                    T_ = T_ @ T
                    odom.append(T_)
        return odom


if __name__ == '__main__':
    euroc = EuRoC()
