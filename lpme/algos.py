import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed


def rand_resp_sum(bit_l, eps):
    msum = np.sum([bit if np.random.uniform(0., 1.) < np.exp(eps)/(1.+np.exp(eps)) else 1.-bit for bit in bit_l])
    return ((np.exp(eps)+1.)/(np.exp(eps)-1.) * msum - len(bit_l) / (np.exp(eps)-1.))/len(bit_l)


def bin_quant(x_l, target_p, eps, q_min, q_max, lamb, T):
    n = len(x_l)
    part_size = n // T
    left = q_min
    right = q_max
    for j in range(T):
        mid = (left + right) / 2.
        start = j*part_size
        if j == (T-1):
            end = n
        else:
            end = (j+1)*part_size
        part_x_l = x_l[start:end]
        noisy_count = rand_resp_sum([float(x < mid) for x in part_x_l], eps)
        if noisy_count > (target_p + lamb/2.):
            right = mid
        elif noisy_count < (target_p - lamb/2.):
            left = mid
        else:
            break
    return mid


def known_var(x_l, sigma, beta, eps, delta, R):
    u = int(2.*R/sigma + 1)
    n = len(x_l)
    #n1 = int( 800 * np.square( (np.exp(eps/2.)+1)/(np.exp(eps/2.)-1) ) * np.log(8.*u/beta) )
    n1 = n // 2
    max_c = -10000
    max_left = None
    max_right = None
    x_l_1 = x_l[:n1]
    x_l_2 = x_l[n1:]
    for j in range(u):
        size = (2*R)/u
        left = -R + j * size
        right = -R + (j+1) * size
        noisy_count = rand_resp_sum([float(left <= x <= right) for x in x_l_1], eps)
        if noisy_count > max_c:
            max_c = noisy_count
            max_left = left
            max_right = right
    mid = (max_left+max_right)/2.
    st = 2 * sigma + sigma * np.sqrt(2 * np.log(8*n/beta))
    res = np.mean([min(mid+st, max(mid-st, x)) for x in x_l_2]) + np.random.normal(0, 2*st/eps/np.sqrt(n-n1))
    return res


def unk_var(x_l, sigma_min, sigma_max, beta, eps, delta, R):
    n = len(x_l)
    #print('n ', n)
    T_med = int(np.log(8*R/sigma_min))
    '''
    n1 = T_med/np.square(0.098) * np.square( (np.exp(eps/2.)+1)/(np.exp(eps/2.)-1) ) * np.log(16.*T_med/beta)
    print('n1 ', n1, 'T_med ', T_med, 'np.square( (np.exp(eps/2.)+1)/(np.exp(eps/2.)-1) ) ', np.square( (np.exp(eps/2.)+1)/(np.exp(eps/2.)-1) ), 'np.log(16.*T_med/beta) ', np.log(16.*T_med/beta))
    '''
    t_mu = bin_quant(x_l, 0.5, eps/np.sqrt(3.), -R, R, 0.098, T_med)

    T_sd = int(np.log((8*R+4*sigma_max)/sigma_min))
    '''
    n2 = T_sd/np.square(0.052) * np.square( (np.exp(eps/2.)+1)/(np.exp(eps/2.)-1) ) * np.log(16.*T_sd/beta)
    print('n2 ', n2)
    '''
    t_sigma = bin_quant(x_l, 0.8413, eps/np.sqrt(3.), -R, R, 0.052, T_sd)
    st = max((t_sigma - t_mu), 0) * (0.5 + 2 * np.sqrt(2 * np.log(8*n/beta)))
    res = np.mean([min(t_mu+st, max(t_mu-st, x)) for x in x_l]) + np.random.normal(0., 2*st/eps*np.sqrt(3.)/np.sqrt(n))
    return res


def unk_var_wrapper(args):
    return unk_var(*args)

def high_dim_unk_var(x, sigma_min, sigma_max, beta, eps, delta, R, para='0'):
    d = x.shape[1]
    if para=='0':
        num_cores = mp.cpu_count()
        res_l = Parallel(n_jobs=num_cores)(delayed(unk_var)(list(x[:,o]), sigma_min, sigma_max, beta, eps/np.sqrt(d), delta, R) for o in range(d))
    elif para=='1':
        with mp.Pool(4) as pool:
            res_l = pool.map(unk_var_wrapper, [(list(x[:,o]), sigma_min, sigma_max, beta, eps/np.sqrt(d), delta, R) for o in range(d)])
    else:
        res_l = []
        for i in range(d):
            res_l.append(unk_var(list(x[:, i]), sigma_min, sigma_max, beta, eps/np.sqrt(d), delta, R))
    return np.array(res_l)


def known_var_wrapper(args):
    return known_var(*args)

def high_dim_known_var(x, sigma, beta, eps, delta, R, para='0'):
    d = x.shape[1]
    if para=='0':
        num_cores = mp.cpu_count()
        res_l = Parallel(n_jobs=num_cores)(delayed(known_var)(list(x[:,o]), sigma, beta, eps/np.sqrt(d), delta, R) for o in range(d))
    elif para=='1':
        with mp.Pool(4) as pool:
            res_l = pool.map(known_var_wrapper, [(list(x[:,o]), sigma, beta, eps/np.sqrt(d), delta, R) for o in range(d)])
    else:
        res_l = []
        for i in range(d):
            res_l.append(known_var(list(x[:, i]), sigma, beta, eps/np.sqrt(d), delta, R))
    return np.array(res_l)
