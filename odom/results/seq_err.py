if __name__ == '__main__':
    import numpy as np
    from sys import argv
    bdir = argv[1]
    tfns = ['{:02d}_tl.txt'.format(i) for i in range(11)]
    rfns = ['{:02d}_rl.txt'.format(i) for i in range(11)]
    terr = []
    rerr = []
    c = 180.0 / np.pi
    for fn in tfns:
        with open(f'{bdir}/{fn}', 'r') as f:
            lines = f.read().split('\n')
        lines = [l.split(' ') for l in lines]
        lines = [l for l in lines if len(l) == 2]
        terr.append(np.mean([float(l[1]) for l in lines]))
    for fn in rfns:
        with open(f'{bdir}/{fn}', 'r') as f:
            lines = f.read().split('\n')
        lines = [l.split(' ') for l in lines]
        lines = [l for l in lines if len(l) == 2]
        rerr.append(np.mean([c * float(l[1]) for l in lines]))
    #print(terr)
    #print(rerr)
    with open(f'{bdir}/by_seq.txt', 'w') as f:
        seqs = ['{:02d}'.format(i) for i in range(11)]
        #print(seqs)
        f.write('seq,tr (%),rot (deg/m)\n')
        for i, s in enumerate(seqs):
            #f.write(f'{s} & {terr[i]} & {rerr[i]}\n')
            f.write('{} & {:.2f} & {:.4f}\n'.format(s, 100 * terr[i], rerr[i]))
