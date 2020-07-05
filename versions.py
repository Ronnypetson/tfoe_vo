import os


def files_to_str(fns):
    s = ''
    for fn in fns:
        with open(fn, 'r') as f:
            s += f.read()
    return s


def save_state(dir, fns):
    '''
    dir is the top directory of experiments
    fns are the names of files going to be saved as state
    '''
    if not os.path.isdir(dir):
        os.makedirs(dir)
    h = hash(files_to_str(fns))
    h = str(h)
    vdir = f'{dir}/{h}/config/'
    if not os.path.isdir(vdir):
        os.makedirs(vdir)
    for fn in fns:
        bfn = os.path.basename(fn)
        dest = f'{vdir}/{bfn}'
        if not os.path.isfile(dest):
            cmd = f'cp {fn} {dest}'
            os.system(cmd)
        else:
            assert hash(files_to_str([dest])) == hash(files_to_str([fn]))
