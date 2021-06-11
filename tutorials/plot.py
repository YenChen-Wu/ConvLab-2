import sys
sys.path.append('/scratch/ycw30/dialtool')
import matplotlib.pyplot as plt
import glob
import numpy as np
from datetime import datetime


def get_log(ep, stage='mle', key='success', mode=None):
    '''
    return: list of float or str
    '''
    
    def get_seed(fname):

        if not fname:
            return None
        if len(fname)==1:
            fname =fname[0]
        
        with open(fname) as f:
            lines = f.readlines()
            if key == 'success':
                ans = [ line.split(' ')[10] for line in lines if 'All' in line ]
            elif key == 'rewards':
                ans = [ line.split(' ')[11] for line in lines if 'total avg reward' in line ]
            else:
                print('The key should be success or rewards')
        return ans[0] if ans else None
    
    if ep=='benchmark':
        log = [ '{}: {}'.format(f[14:-4].upper(), get_seed(f)) for f in glob.glob('log/benchmark*')]
        for i in log:
            print(i.strip())
        return
    
    else:
        n_seed = len(glob.glob('log/'+stage+'/*'))
        log = [ get_seed(glob.glob('log/'+stage+'/'+str(s)+'/*_'+str(ep)+'.log')) for s in range(n_seed) ]
        
        return log

def get_loss(exp,key='v'):
    
    def get_seed(fname):
        with open(fname) as f:
            lines = f.readlines()
            if key=='v':
#                 print ('value loss')
                loss = [ line.split(' ')[9] for line in lines if 'value' in line ]
            elif key=='p':
#                 print ('policy loss')
                loss = [ line.split(' ')[9] for line in lines if 'policy,' in line ]
            elif key=='adv_mean':
                loss = [ line.split(' ')[5][:-1] for line in lines if 'Adv' in line ]
                return np.asarray(loss,dtype=float)
            elif key=='adv_std':
                loss = [ line.split(' ')[7] for line in lines if 'Adv' in line ]
                return np.asarray(loss,dtype=float)
            elif key=='v_mean':
                loss = [ line.split(' ')[5][:-1] for line in lines if 'Value' in line ]
                return np.asarray(loss,dtype=float)
            elif key=='v_std':
                loss = [ line.split(' ')[7] for line in lines if 'Value' in line ]
                return np.asarray(loss,dtype=float)
        return np.asarray(loss,dtype=float).reshape((-1,5)).mean(axis=1)

    model = exp.split('_')[0]
    D = np.asarray([ get_seed(f) for f in glob.glob('convlab2/policy/'+model+'/log/log_'+exp+'*')]).transpose()
    x = list(range(len(D)))
    
    return x,D


