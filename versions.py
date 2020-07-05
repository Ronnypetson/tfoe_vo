import os
import hashlib


def files_to_hash(fns):
    h = 0
    for fn in fns:
        with open(fn, 'r') as f:
            content = f'{f.read()}'.encode('utf-8')
            h_ = hashlib.sha224(content).hexdigest()
            h += int(h_, 16)
    return h


def save_state(dir, fns):
    '''
    dir is the top directory of experiments
    fns are the names of files going to be saved as state
    '''
    if not os.path.isdir(dir):
        os.makedirs(dir)
    #h = hash(files_to_str(fns))
    h = files_to_hash(fns)
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
            assert files_to_hash([dest]) == files_to_hash([fn])
