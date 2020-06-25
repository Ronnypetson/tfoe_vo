import numpy as np
import pykitti

def load_kitti_odom(basedir,s,start,end):
    assert end > start
    data = pykitti.odometry(basedir,s)
    odom = data.poses[start:end]
    inert = np.linalg.inv(odom[0])
    odom = [inert @ p for p in odom]
    odom = np.array(odom)
    return odom

if __name__ == '__main__':
    basedir = '/home/ronnypetson/Downloads/kitti_seq/dataset/'
    odom = load_kitti_odom(basedir,'01',0,50)
    print(odom.shape)

