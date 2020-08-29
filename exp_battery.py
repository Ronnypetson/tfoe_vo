import os


if __name__ == '__main__':
    seq = ['04', '03', '01', '06', '07', '10', '09', '05', '08', '00', '02']
    for s in seq[:3]:
        print(f'Starting evaluation of sequence {s}')
        #os.system(f'python3 kitti_slocal.py {s} eval odom/results/BA_eval_w3_g2/data/')
        os.system(f'python3 kitti_slocal.py {s} init odom/results/BA_init_w3_g1/data/')
