import numpy as np
import time
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def clipped_mean(x, p, qt=None, m_l=None):
    p1 = p * 0.25
    p2 = p * 0.75
    n = x.shape[0]
    d = x.shape[1]
    x_norm = np.linalg.norm(x, axis=1)
    if qt is None:
        m = int(n - 2.*np.sqrt(d/(2.*p2)))
    else:
        m = int(qt)
    C = sorted(x_norm)[m]
    x_clipped = []
    for i in range(len(x)):
        xi_norm = x_norm[i]
        scale = min(C/xi_norm,1.0)
        x_clipped.append(scale*np.array(x[i]))
    mean = np.mean(x_clipped,axis=0)
    noisy_mean = mean + np.random.normal(0, 2.*C/np.sqrt(2.*p2)/n, size=d)
    return noisy_mean



n_trials = 30
n = 500
p = 0.1
d_l = [128, 512, 2048, 8192, 8192*4]


terr_ours_l = []
terr_95_l = []
terr_85_l = []
terr_75_l = []
terr_50_l = []

for d in d_l:
    err_ours_l = []
    err_95_l = []
    err_85_l = []
    err_75_l = []
    err_50_l = []
    for _ in range(n_trials):
        #norm = np.random.randint(1, n+1, size=(n, 1))
        norm = np.expand_dims(np.arange(n)+1, 1)
        x = np.ones((n, d)) * np.square(norm)
        #x = np.random.normal(0, 1, size=(n, d))
        mean_gt = np.mean(x, axis=0)
        mean_ours = clipped_mean(x, p)
        mean_95 = clipped_mean(x, p, 0.95*n)
        mean_85 = clipped_mean(x, p, 0.85*n)
        mean_75 = clipped_mean(x, p, 0.75*n)
        mean_50 = clipped_mean(x, p, 0.5*n)

        err_ours_l.append(np.linalg.norm(mean_ours-mean_gt))
        err_95_l.append(np.linalg.norm(mean_95-mean_gt))
        err_85_l.append(np.linalg.norm(mean_85-mean_gt))
        err_75_l.append(np.linalg.norm(mean_75-mean_gt))
        err_50_l.append(np.linalg.norm(mean_50-mean_gt))

    terr_ours_l.append(np.mean(err_ours_l))
    terr_95_l.append(np.mean(err_95_l))
    terr_85_l.append(np.mean(err_85_l))
    terr_75_l.append(np.mean(err_75_l))
    terr_50_l.append(np.mean(err_50_l))

    '''
    terr_ours_l.append(stats.trim_mean(err_ours_l, 0.1))
    terr_95_l.append(stats.trim_mean(err_95_l, 0.1))
    terr_85_l.append(stats.trim_mean(err_85_l, 0.1))
    terr_75_l.append(stats.trim_mean(err_75_l, 0.1))
    terr_50_l.append(stats.trim_mean(err_50_l, 0.1))
    '''

plt.plot(d_l, terr_50_l, color='red', marker='+', label=r'50%')
plt.plot(d_l, terr_75_l, color='magenta', marker='+', label=r'75%')
plt.plot(d_l, terr_85_l, color='gray', marker='+', label=r'85%')
plt.plot(d_l, terr_95_l, color='blue', marker='+', label=r'95%')
plt.plot(d_l, terr_ours_l, color='green', marker='*', label='Ours')

plt.legend(loc='lower right', framealpha=0.4)
plt.xlim((92, 8192*6))
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel(r'$d$',fontsize=16)
plt.ylabel(r'$\ell_2$ error',fontsize=16)
plt.grid()
plt.savefig('./syn_qt.pdf',bbox_inches='tight', pad_inches=0)
plt.clf()


