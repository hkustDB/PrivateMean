import numpy as np
import torch
from scipy.stats import trim_mean
from coinpress.algos import L2, multivariate_mean_iterative, cov_est
from coinpress.utils import parse_args
from lpme.algos import high_dim_known_var, high_dim_unk_var
from quantile_binary_search.method import random_rotation_mean, clipped_mean
import quantile_binary_search.ldpmethod as ldp
from utils import get_mnist_data, get_mnist_labels, load_paths, save_paths, write_output, write_text


def coinpress_mean(X, c, r, t, p, func=multivariate_mean_iterative):
    if t==1:
        Ps = [p]
    else:
        Ps = [(1.0/4.0/(t-1))*p for i in range(t-1)]
        Ps.append((3.0/4.0)*p)
    mean = func(X.copy(), c, r, t, Ps)
    return mean

def mat_inv_sqrt(Sigma):
    Sigma_inv = torch.inverse(Sigma)
    U_inv, D_inv, V_inv = Sigma_inv.svd()
    Sigma_inv_sqrt = torch.mm(U_inv, torch.mm(D_inv.sqrt().diag_embed(), V_inv.t()))

    U, D, V = Sigma.svd()
    Sigma_sqrt = torch.mm(U ,torch.mm(D.sqrt().diag_embed(), V.t()))
    return Sigma_inv_sqrt, Sigma_sqrt

def mahalanobis_dist(vec, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    U_inv, D_inv, V_inv = np.linalg.svd(Sigma_inv)
    Sigma_inv_sqrt = np.matmul(np.matmul(U_inv,np.diag(np.sqrt(D_inv))),V_inv) 
    vec_normalized = np.matmul(Sigma_inv_sqrt, vec)
    return np.linalg.norm(vec_normalized)

def coinpress_cov_mean(X, c, r, t, p, d, tmean=10):
    p0 = 0.5*p
    if t==1:
        Ps = [p0]
    else:
        Ps = [(1.0/4.0/(t-1))*p0 for i in range(t-1)]
        Ps.append((3.0/4.0)*p0)
    pargs = parse_args()
    pargs.rho = Ps
    pargs.t = t
    pargs.d = d
    pargs.u = r/np.sqrt(d)
    X_tens = torch.from_numpy(X.astype(np.float32))
    cov_pr = cov_est(X_tens,pargs)
    mat_inv, mat_sqrt = mat_inv_sqrt(cov_pr)
    X_scaled = torch.mm(X_tens,mat_inv)
    mean_cp = coinpress_mean(X_scaled.numpy(),c,r,tmean,p0)
    mean = np.matmul(mean_cp,mat_sqrt.numpy())
    return mean

def generate_gauss_paths(mean,cov,nPaths,nSamples,d):
    X = []
    for i in range(nPaths):
        X.append(np.random.multivariate_normal(mean, cov, int(nSamples)))
    return X
    
def generate_laplace_paths(mean,cov,nPaths,nSamples,d):
    X = []
    scale = np.sqrt(cov[0,0]/2)
    for i in range(nPaths):
        X.append(np.random.laplace(mean[0], scale, (int(nSamples),d)))
    return X

def cov_id(sigma0,d):
    cov = sigma0*sigma0*np.eye(d)
    return cov

def cov_sigma(sigma0,d):
    X = np.random.multivariate_normal([0]*d, np.eye(d), 1000)
    U = np.random.uniform(low=[0]*d,high=[1.0]*d,size=(d,d))
    X = np.matmul(X,U)
    cov0 = np.matmul(np.transpose(X),X)
    V1, S, V2 = np.linalg.svd(cov0)
    diag_vec = np.random.uniform(low=1,high=sigma0,size=d)
    cov = np.matmul(np.matmul(V1,np.diag(diag_vec)),V2)
    return cov

def get_mnist_digit(dig=0,b_comb=True):
    train_data, test_data = get_mnist_data()
    train_labels, test_labels = get_mnist_labels()
    if b_comb:
        data = np.concatenate([train_data,test_data],axis=0)
        labels = np.concatenate([train_labels,test_labels],axis=0)
    else:
        data = train_data.copy()
        labels = train_labels.copy()
    inds = (labels==dig)
    return data[inds]
    
def generate_mnist_paths(dig, nSamples=6000):
    if nSamples > 6000:
        data = get_mnist_digit(dig,b_comb=True)
    else:
        data = get_mnist_digit(dig,b_comb=False)
    return data
    



def test_dist(mean,cov,p,nPaths,nSamples,d,c,r,u,bGenerate=True,generate_func=generate_gauss_paths,prefix='gauss_',folder=''):
    if bGenerate:
        X_paths = generate_func(mean,cov,nPaths,nSamples,d)
        if not(folder==''):
            filename=prefix+str(nPaths)+'paths_'+str(nSamples)+'samples_dim'+str(d)+'.txt'
            save_paths(X_paths,nPaths,nSamples,d,folder,filename)
    else:
        filename=prefix+str(nPaths)+'paths_'+str(nSamples)+'samples_dim'+str(d)+'.txt'
        X_paths = load_paths(folder+filename,nPaths,nSamples,d)
    non_pr = []
    means_t1 = []
    means_t2 = []
    means_t3 = []
    means_t4 = []
    means_t10 = []
    means_hada = []
    means_clip = []
    for i in range(nPaths):
        X = X_paths[i]
        non_pr.append(L2(np.mean(X, axis=0)-mean))
        means_hada.append(L2(random_rotation_mean(X.copy(),d,u,p)-mean))
        means_clip.append(L2(clipped_mean(X.copy(),nSamples,d,u,p)-mean))
        means_t1.append(L2(coinpress_mean(X.copy(), c, r, 1, p, func=multivariate_mean_iterative)-mean))
        means_t2.append(L2(coinpress_mean(X.copy(), c, r, 2, p, func=multivariate_mean_iterative)-mean))
        means_t3.append(L2(coinpress_mean(X.copy(), c, r, 3, p, func=multivariate_mean_iterative)-mean))
        means_t4.append(L2(coinpress_mean(X.copy(), c, r, 4, p, func=multivariate_mean_iterative)-mean))
        means_t10.append(L2(coinpress_mean(X.copy(), c, r, 10, p, func=multivariate_mean_iterative)-mean))
    return non_pr, means_t1, means_t2,  means_t3, means_t4, means_t10, means_hada, means_clip


def test_dist_cov(mean,cov,p,nPaths,nSamples,d,c,r,u,bGenerate=True,generate_func=generate_gauss_paths,prefix='gauss_',folder=''):
    if bGenerate:
        X_paths = generate_func(mean,cov,nPaths,nSamples,d)
        if not(folder==''):
            filename=prefix+str(nPaths)+'paths_'+str(nSamples)+'samples_dim'+str(d)+'.txt'
            save_paths(X_paths,nPaths,nSamples,d,folder,filename)
    else:
        filename=prefix+str(nPaths)+'paths_'+str(nSamples)+'samples_dim'+str(d)+'.txt'
        X_paths = load_paths(folder+filename,nPaths,nSamples,d)
    non_pr = []
    l2_t1 = []
    l2_t2 = []
    l2_t3 = []
    l2_t4 = []
    l2_t5 = []
    l2_hada = []

    maha_nonpr = []
    maha_t1 = []
    maha_t2 = []
    maha_t3 = []
    maha_t4 = []
    maha_t5 = []
    maha_hada = []
    for i in range(nPaths):
        X = X_paths[i]
        err_nonpr = np.mean(X, axis=0)-mean
        non_pr.append(L2(err_nonpr))
        maha_nonpr.append(mahalanobis_dist(err_nonpr,cov))

        err_hada = random_rotation_mean(X.copy(),d,u,p)-mean        
        l2_hada.append(L2(err_hada))
        maha_hada.append(mahalanobis_dist(err_hada,cov))

        err_t1 = coinpress_cov_mean(X.copy(), c, r, 1, p, d)-mean
        l2_t1.append(L2(err_t1))
        maha_t1.append(mahalanobis_dist(err_t1,cov))

        err_t2 = coinpress_cov_mean(X.copy(), c, r, 2, p, d)-mean
        l2_t2.append(L2(err_t2))
        maha_t2.append(mahalanobis_dist(err_t2,cov))

        err_t3 = coinpress_cov_mean(X.copy(), c, r, 3, p, d)-mean
        l2_t3.append(L2(err_t3))
        maha_t3.append(mahalanobis_dist(err_t3,cov))

        err_t4 = coinpress_cov_mean(X.copy(), c, r, 4, p, d)-mean
        l2_t4.append(L2(err_t4))
        maha_t4.append(mahalanobis_dist(err_t4,cov))

        err_t5 = coinpress_cov_mean(X.copy(), c, r, 5, p, d)-mean
        l2_t5.append(L2(err_t5))
        maha_t5.append(mahalanobis_dist(err_t5,cov))
    return (non_pr,maha_nonpr), (l2_t1,maha_t1), (l2_t2,maha_t2),  (l2_t3,maha_t3), (l2_t4,maha_t4), (l2_t5,maha_t5), (l2_hada,maha_hada)


def pad_data(x):
    n = x.shape[0]
    d = x.shape[1]
    d_pad = 2**int(np.ceil(np.log2(d)))
    X_pad = np.zeros((n,d_pad))
    X_pad[:,:d] = x
    return d, X_pad, d_pad


def test_emp(p,nPaths,nSamples,r,u,generate_func):
    X = generate_func(nSamples)
    d, X_pad, d_pad = pad_data(X)
    c = [0.0]*d
    means_t1 = []
    means_t2 = []
    means_t3 = []
    means_t4 = []
    means_t10 = []
    means_hada = []
    mean = np.mean(X, axis=0)
    for i in range(nPaths):
        means_hada.append(L2(random_rotation_mean(X_pad.copy(),d_pad,u,p)[:d]-mean))
        means_t1.append(L2(coinpress_cov_mean(X.copy(), c, r, 1, p, func=multivariate_mean_iterative)-mean))
        means_t2.append(L2(coinpress_mean(X.copy(), c, r, 2, p, func=multivariate_mean_iterative)-mean))
        means_t3.append(L2(coinpress_mean(X.copy(), c, r, 3, p, func=multivariate_mean_iterative)-mean))
        means_t4.append(L2(coinpress_mean(X.copy(), c, r, 4, p, func=multivariate_mean_iterative)-mean))
        means_t10.append(L2(coinpress_mean(X.copy(), c, r, 10, p, func=multivariate_mean_iterative)-mean))
    return means_t1, means_t2,  means_t3, means_t4, means_t10, means_hada

def test_emp_cov(p,nPaths,nSamples,r,u,generate_func):
    X = generate_func(nSamples)
    d, X_pad, d_pad = pad_data(X)
    c = [0.0]*d
    l2_t1 = []
    l2_t2 = []
    l2_t3 = []
    l2_t4 = []
    l2_t5 = []
    l2_hada = []
    mean = np.mean(X, axis=0)
    for i in range(nPaths):
        err_hada = random_rotation_mean(X_pad.copy(),d_pad,u,p)[:d]-mean    
        l2_hada.append(L2(err_hada))

        err_t1 = coinpress_cov_mean(X.copy(), c, r, 1, p, d)-mean
        l2_t1.append(L2(err_t1))

        err_t2 = coinpress_cov_mean(X.copy(), c, r, 2, p, d)-mean
        l2_t2.append(L2(err_t2))

        err_t3 = coinpress_cov_mean(X.copy(), c, r, 3, p, d)-mean
        l2_t3.append(L2(err_t3))

        err_t4 = coinpress_cov_mean(X.copy(), c, r, 4, p, d)-mean
        l2_t4.append(L2(err_t4))

        err_t5 = coinpress_cov_mean(X.copy(), c, r, 5, p, d)-mean
        l2_t5.append(L2(err_t5))
    return l2_t1, l2_t2, l2_t3, l2_t4, l2_t5, l2_hada



def test_dim(mean0,sigma0,dims,p,nPaths,nSamples,rfact=50,T=10,bGenerate=True,folder='',generate_func=generate_gauss_paths,cov_func=cov_id,prefix=''):
    err_nonpr = []
    err_t1 = []
    err_t2 = []
    err_t3 = []
    err_t4 = []
    err_t10 = []
    err_hada = []
    err_clip = []

    err_nonpr_paths = []
    err_t1_paths = []
    err_t2_paths = []
    err_t3_paths = []
    err_t4_paths = []
    err_t10_paths = []
    err_hada_paths = []
    err_clip_paths = []

    for d in dims:
        print('d=',d)
        mean = [mean0]*d
        cov = cov_func(sigma0,d)
        c = [0.0]*d
        r = rfact*np.sqrt(d)
        u = 2.0*r
        assert(np.linalg.norm(np.array(mean)-np.array(c))<=r)
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/np.sqrt(2.0*p))
        non_pr, mean_t1, mean_t2, mean_t3, mean_t4, mean_t10, mean_hada, mean_clip = test_dist(mean,cov,p,nPaths,nSamples,d,c,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr,0.1))
        err_t1.append(trim_mean(mean_t1,0.1))
        err_t2.append(trim_mean(mean_t2,0.1))
        err_t3.append(trim_mean(mean_t3,0.1))
        err_t4.append(trim_mean(mean_t4,0.1))
        err_t10.append(trim_mean(mean_t10,0.1))
        err_hada.append(trim_mean(mean_hada,0.1))
        err_clip.append(trim_mean(mean_clip,0.1))

        err_nonpr_paths.append(non_pr)
        err_t1_paths.append(mean_t1)
        err_t2_paths.append(mean_t2)
        err_t3_paths.append(mean_t3)
        err_t4_paths.append(mean_t4)
        err_t10_paths.append(mean_t10)
        err_hada_paths.append(mean_hada)
        err_clip_paths.append(mean_clip)

    strdims = '|'.join([str(di) for di in dims])
    params = [['nPaths','nSamples','d','mean','sigma','p','u','r'],[str(nPaths),str(nSamples),strdims,str(mean0),str(sigma0),str(p),str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_d' + str(dims[0]) + '-' +str(dims[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(dims,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(dims,err_t1,strfolder,'err_t1.txt')
    write_output(dims,err_t2,strfolder,'err_t2.txt')
    write_output(dims,err_t3,strfolder,'err_t3.txt')
    write_output(dims,err_t4,strfolder,'err_t4.txt')
    write_output(dims,err_t10,strfolder,'err_t10.txt')
    write_output(dims,err_hada,strfolder,'err_hada.txt')
    write_output(dims,err_clip,strfolder,'err_clip.txt')

    write_output(dims,err_nonpr_paths,strfolder,'err_nonpr_paths.txt')
    write_output(dims,err_t1_paths,strfolder,'err_t1_paths.txt')
    write_output(dims,err_t2_paths,strfolder,'err_t2_paths.txt')
    write_output(dims,err_t3_paths,strfolder,'err_t3_paths.txt')
    write_output(dims,err_t4_paths,strfolder,'err_t4_paths.txt')
    write_output(dims,err_t10_paths,strfolder,'err_t10_paths.txt')
    write_output(dims,err_hada_paths,strfolder,'err_hada_paths.txt')
    write_output(dims,err_clip_paths,strfolder,'err_clip_paths.txt')
    return subfolder

def test_dim_cov(mean0,sigma0,dims,p,nPaths,nSamples,rfact=50,T=10,bGenerate=True,folder='',generate_func=generate_gauss_paths,cov_func=cov_id,prefix=''):
    err_nonpr = []
    err_t1 = []
    err_t2 = []
    err_t3 = []
    err_t4 = []
    err_t5 = []
    err_hada = []

    maha_nonpr = []
    maha_t1 = []
    maha_t2 = []
    maha_t3 = []
    maha_t4 = []
    maha_t5 = []
    maha_hada = []

    for d in dims:
        non_pr = []
        print('d=',d)
        mean = [mean0]*d
        cov = cov_func(sigma0,d)
        c = [0.0]*d
        r = rfact*np.sqrt(d)
        u = 2.0*r
        assert(np.linalg.norm(np.array(mean)-np.array(c))<=r)
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/np.sqrt(2.0*p))
        non_pr, mean_t1, mean_t2, mean_t3, mean_t4, mean_t5, mean_hada = test_dist_cov(mean,cov,p,nPaths,nSamples,d,c,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr[0],0.1))
        err_t1.append(trim_mean(mean_t1[0],0.1))
        err_t2.append(trim_mean(mean_t2[0],0.1))
        err_t3.append(trim_mean(mean_t3[0],0.1))
        err_t4.append(trim_mean(mean_t4[0],0.1))
        err_t5.append(trim_mean(mean_t5[0],0.1))
        err_hada.append(trim_mean(mean_hada[0],0.1))

        maha_nonpr.append(trim_mean(non_pr[1],0.1))
        maha_t1.append(trim_mean(mean_t1[1],0.1))
        maha_t2.append(trim_mean(mean_t2[1],0.1))
        maha_t3.append(trim_mean(mean_t3[1],0.1))
        maha_t4.append(trim_mean(mean_t4[1],0.1))
        maha_t5.append(trim_mean(mean_t5[1],0.1))
        maha_hada.append(trim_mean(mean_hada[1],0.1))

    strdims = '|'.join([str(di) for di in dims])
    params = [['nPaths','nSamples','d','mean','sigma','p','u','r'],[str(nPaths),str(nSamples),strdims,str(mean0),str(sigma0),str(p),str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_d' + str(dims[0]) + '-' +str(dims[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(dims,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(dims,err_t1,strfolder,'err_t1.txt')
    write_output(dims,err_t2,strfolder,'err_t2.txt')
    write_output(dims,err_t3,strfolder,'err_t3.txt')
    write_output(dims,err_t4,strfolder,'err_t4.txt')
    write_output(dims,err_t5,strfolder,'err_t5.txt')
    write_output(dims,err_hada,strfolder,'err_hada.txt')

    write_output(dims,maha_nonpr,strfolder,'maha_nonpr.txt')
    write_output(dims,maha_t1,strfolder,'maha_t1.txt')
    write_output(dims,maha_t2,strfolder,'maha_t2.txt')
    write_output(dims,maha_t3,strfolder,'maha_t3.txt')
    write_output(dims,maha_t4,strfolder,'maha_t4.txt')
    write_output(dims,maha_t5,strfolder,'maha_t5.txt')
    write_output(dims,maha_hada,strfolder,'maha_hada.txt')
    return subfolder

def test_rho(mean0,sigma0,d,ps,nPaths,nSamples,rfact=50,T=10,bGenerate=True,folder='',generate_func=generate_gauss_paths,prefix=''):
    err_nonpr = []
    err_t1 = []
    err_t2 = []
    err_t3 = []
    err_t4 = []
    err_t10 = []
    err_hada = []
    err_clip = []

    err_nonpr_paths = []
    err_t1_paths = []
    err_t2_paths = []
    err_t3_paths = []
    err_t4_paths = []
    err_t10_paths = []
    err_hada_paths = []
    err_clip_paths = []

    mean = [mean0]*d
    cov = sigma0*sigma0*np.eye(d)
    c = [0.0]*d
    r = rfact*np.sqrt(d)
    u = 2.0*r
    for p in ps:
        print('rho = ',p)
        assert(np.linalg.norm(np.array(mean)-np.array(c))<=r)
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/np.sqrt(2.0*p))
        non_pr, mean_t1, mean_t2, mean_t3, mean_t4, mean_t10, mean_hada, mean_clip = test_dist(mean,cov,p,nPaths,nSamples,d,c,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr,0.1))
        err_t1.append(trim_mean(mean_t1,0.1))
        err_t2.append(trim_mean(mean_t2,0.1))
        err_t3.append(trim_mean(mean_t3,0.1))
        err_t4.append(trim_mean(mean_t4,0.1))
        err_t10.append(trim_mean(mean_t10,0.1))
        err_hada.append(trim_mean(mean_hada,0.1))
        err_clip.append(trim_mean(mean_clip,0.1))

        err_nonpr_paths.append(non_pr)
        err_t1_paths.append(mean_t1)
        err_t2_paths.append(mean_t2)
        err_t3_paths.append(mean_t3)
        err_t4_paths.append(mean_t4)
        err_t10_paths.append(mean_t10)
        err_hada_paths.append(mean_hada)
        err_clip_paths.append(mean_clip)

    strps = '|'.join([str(pi) for pi in ps])
    params = [['nPaths','nSamples','d','mean','sigma','p','u','r'],[str(nPaths),str(nSamples),str(d),str(mean0),str(sigma0),strps,str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_rho' +str(ps[0]) +'-' + str(ps[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(ps,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(ps,err_t1,strfolder,'err_t1.txt')
    write_output(ps,err_t2,strfolder,'err_t2.txt')
    write_output(ps,err_t3,strfolder,'err_t3.txt')
    write_output(ps,err_t4,strfolder,'err_t4.txt')
    write_output(ps,err_t10,strfolder,'err_t10.txt')
    write_output(ps,err_hada,strfolder,'err_hada.txt')
    write_output(ps,err_clip,strfolder,'err_clip.txt')

    write_output(ps,err_nonpr_paths,strfolder,'err_nonpr_paths.txt')
    write_output(ps,err_t1_paths,strfolder,'err_t1_paths.txt')
    write_output(ps,err_t2_paths,strfolder,'err_t2_paths.txt')
    write_output(ps,err_t3_paths,strfolder,'err_t3_paths.txt')
    write_output(ps,err_t4_paths,strfolder,'err_t4_paths.txt')
    write_output(ps,err_t10_paths,strfolder,'err_t10_paths.txt')
    write_output(ps,err_hada_paths,strfolder,'err_hada_paths.txt')
    write_output(ps,err_clip_paths,strfolder,'err_clip_paths.txt')
    return subfolder


def test_rho_emp(d,ps,nPaths,nSamples,rfact=50,T=10,folder='',generate_func=generate_mnist_paths,prefix=''):
    err_t1 = []
    err_t2 = []
    err_t3 = []
    err_t4 = []
    err_t10 = []
    err_hada = []

    err_t1_paths = []
    err_t2_paths = []
    err_t3_paths = []
    err_t4_paths = []
    err_t10_paths = []
    err_hada_paths = []

    r = rfact*np.sqrt(d)
    u = 2.0*r
    for p in ps:
        print('rho = ',p)
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/np.sqrt(2.0*p))
        mean_t1, mean_t2, mean_t3, mean_t4, mean_t10, mean_hada = test_emp(p,nPaths,nSamples,r,u,generate_func=generate_func)
        err_t1.append(trim_mean(mean_t1,0.1))
        err_t2.append(trim_mean(mean_t2,0.1))
        err_t3.append(trim_mean(mean_t3,0.1))
        err_t4.append(trim_mean(mean_t4,0.1))
        err_t10.append(trim_mean(mean_t10,0.1))
        err_hada.append(trim_mean(mean_hada,0.1))

        err_t1_paths.append(mean_t1)
        err_t2_paths.append(mean_t2)
        err_t3_paths.append(mean_t3)
        err_t4_paths.append(mean_t4)
        err_t10_paths.append(mean_t10)
        err_hada_paths.append(mean_hada)

    strps = '|'.join([str(pi) for pi in ps])
    params = [['nPaths','nSamples','d','p','u','r'],[str(nPaths),str(nSamples),str(d),strps,str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_rho' +str(ps[0]) +'-' + str(ps[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(ps,err_t1,strfolder,'err_t1.txt')
    write_output(ps,err_t2,strfolder,'err_t2.txt')
    write_output(ps,err_t3,strfolder,'err_t3.txt')
    write_output(ps,err_t4,strfolder,'err_t4.txt')
    write_output(ps,err_t10,strfolder,'err_t10.txt')
    write_output(ps,err_hada,strfolder,'err_hada.txt')

    write_output(ps,err_t1_paths,strfolder,'err_t1_paths.txt')
    write_output(ps,err_t2_paths,strfolder,'err_t2_paths.txt')
    write_output(ps,err_t3_paths,strfolder,'err_t3_paths.txt')
    write_output(ps,err_t4_paths,strfolder,'err_t4_paths.txt')
    write_output(ps,err_t10_paths,strfolder,'err_t10_paths.txt')
    write_output(ps,err_hada_paths,strfolder,'err_hada_paths.txt')
    return subfolder

def test_rho_emp_cov(d,ps,nPaths,nSamples,rfact=50,T=10,folder='',generate_func=generate_mnist_paths,prefix=''):
    err_t1 = []
    err_t2 = []
    err_t3 = []
    err_t4 = []
    err_t5 = []
    err_hada = []

    err_t1_paths = []
    err_t2_paths = []
    err_t3_paths = []
    err_t4_paths = []
    err_t5_paths = []
    err_hada_paths = []

    r = rfact*np.sqrt(d)
    u = 2.0*r
    for p in ps:
        print('rho = ',p)
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/np.sqrt(2.0*p))
        mean_t1, mean_t2, mean_t3, mean_t4, mean_t5, mean_hada = test_emp_cov(p,nPaths,nSamples,r,u,generate_func=generate_func)
        err_t1.append(trim_mean(mean_t1,0.1))
        err_t2.append(trim_mean(mean_t2,0.1))
        err_t3.append(trim_mean(mean_t3,0.1))
        err_t4.append(trim_mean(mean_t4,0.1))
        err_t5.append(trim_mean(mean_t5,0.1))
        err_hada.append(trim_mean(mean_hada,0.1))

        err_t1_paths.append(mean_t1)
        err_t2_paths.append(mean_t2)
        err_t3_paths.append(mean_t3)
        err_t4_paths.append(mean_t4)
        err_t5_paths.append(mean_t5)
        err_hada_paths.append(mean_hada)

    strps = '|'.join([str(pi) for pi in ps])
    params = [['nPaths','nSamples','d','p','u','r'],[str(nPaths),str(nSamples),str(d),strps,str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_rho' +str(ps[0]) +'-' + str(ps[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(ps,err_t1,strfolder,'err_t1.txt')
    write_output(ps,err_t2,strfolder,'err_t2.txt')
    write_output(ps,err_t3,strfolder,'err_t3.txt')
    write_output(ps,err_t4,strfolder,'err_t4.txt')
    write_output(ps,err_t5,strfolder,'err_t5.txt')
    write_output(ps,err_hada,strfolder,'err_hada.txt')


    write_output(ps,err_t1_paths,strfolder,'err_t1_paths.txt')
    write_output(ps,err_t2_paths,strfolder,'err_t2_paths.txt')
    write_output(ps,err_t3_paths,strfolder,'err_t3_paths.txt')
    write_output(ps,err_t4_paths,strfolder,'err_t4_paths.txt')
    write_output(ps,err_t5_paths,strfolder,'err_t5_paths.txt')
    write_output(ps,err_hada_paths,strfolder,'err_hada_paths.txt')

    return subfolder


def test_R(mean0,sigma0,d,p,nPaths,nSamples,rfacts,T=10,bGenerate=True,folder='',generate_func=generate_gauss_paths,cov_func=cov_sigma,prefix=''):
    err_nonpr = []
    err_t1 = []
    err_t2 = []
    err_t3 = []
    err_t4 = []
    err_t10 = []
    err_hada = []
    err_clip = []

    err_nonpr_paths = []
    err_t1_paths = []
    err_t2_paths = []
    err_t3_paths = []
    err_t4_paths = []
    err_t10_paths = []
    err_hada_paths = []
    err_clip_paths = []

    mean = [mean0]*d
    cov = cov_func(sigma0,d)
    c = [0.0]*d
    Rs = [rfact*np.sqrt(d) for rfact in rfacts]
    for r in Rs:
        u = 2.0*r
        print('R = ',r)
        assert(np.linalg.norm(np.array(mean)-np.array(c))<=r)
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/np.sqrt(2.0*p))
        non_pr, mean_t1, mean_t2, mean_t3, mean_t4, mean_t10, mean_hada, mean_clip = test_dist(mean,cov,p,nPaths,nSamples,d,c,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr,0.1))
        err_t1.append(trim_mean(mean_t1,0.1))
        err_t2.append(trim_mean(mean_t2,0.1))
        err_t3.append(trim_mean(mean_t3,0.1))
        err_t4.append(trim_mean(mean_t4,0.1))
        err_t10.append(trim_mean(mean_t10,0.1))
        err_hada.append(trim_mean(mean_hada,0.1))
        err_clip.append(trim_mean(mean_clip,0.1))

        err_nonpr_paths.append(non_pr)
        err_t1_paths.append(mean_t1)
        err_t2_paths.append(mean_t2)
        err_t3_paths.append(mean_t3)
        err_t4_paths.append(mean_t4)
        err_t10_paths.append(mean_t10)
        err_hada_paths.append(mean_hada)
        err_clip_paths.append(mean_clip)

    strRs = '|'.join([str(r) for r in Rs])
    params = [['nPaths','nSamples','d','mean','sigma','p','u','r'],[str(nPaths),str(nSamples),str(d),str(mean0),str(sigma0),str(p),str(u),strRs]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_R' +str(rfacts[0]) + '-' +str(rfacts[-1])+'/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(Rs,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(Rs,err_t1,strfolder,'err_t1.txt')
    write_output(Rs,err_t2,strfolder,'err_t2.txt')
    write_output(Rs,err_t3,strfolder,'err_t3.txt')
    write_output(Rs,err_t4,strfolder,'err_t4.txt')
    write_output(Rs,err_t10,strfolder,'err_t10.txt')
    write_output(Rs,err_hada,strfolder,'err_hada.txt')

    write_output(Rs,err_nonpr_paths,strfolder,'err_nonpr_paths.txt')
    write_output(Rs,err_t1_paths,strfolder,'err_t1_paths.txt')
    write_output(Rs,err_t2_paths,strfolder,'err_t2_paths.txt')
    write_output(Rs,err_t3_paths,strfolder,'err_t3_paths.txt')
    write_output(Rs,err_t4_paths,strfolder,'err_t4_paths.txt')
    write_output(Rs,err_t10_paths,strfolder,'err_t10_paths.txt')
    write_output(Rs,err_hada_paths,strfolder,'err_hada_paths.txt')
    return subfolder


########################### local model tests ########################### 

def test_dist_local(mean,cov,eps,sigma,nPaths,nSamples,d,r,u,bGenerate=True,generate_func=generate_gauss_paths,prefix='gauss_',folder=''):
    if bGenerate:
        X_paths = generate_func(mean,cov,nPaths,nSamples,d)
        if not(folder==''):
            filename=prefix+str(nPaths)+'paths_'+str(nSamples)+'samples_dim'+str(d)+'.txt'
            save_paths(X_paths,nPaths,nSamples,d,folder,filename)
    else:
        filename=prefix+str(nPaths)+'paths_'+str(nSamples)+'samples_dim'+str(d)+'.txt'
        X_paths = load_paths(folder+filename,nPaths,nSamples,d)
    non_pr = []
    err_unk_var = []
    err_known_var = []

    err_hada = []

    p = eps*eps*0.5
    for i in range(nPaths):
        X = X_paths[i]
        non_pr.append(np.linalg.norm(np.mean(X, axis=0)-mean))
        mean_hada = ldp.random_rotation_mean(X.copy(),d,u,p)
        err_hada.append(np.linalg.norm(mean_hada-mean))

        unk_var_mean = high_dim_unk_var(X.copy(), 0.1, r, 0.01, eps, 0.0000001, r)
        err_unk_var.append(np.linalg.norm(unk_var_mean-mean))
        
        unk_var_mean = high_dim_known_var(X.copy(), sigma, 0.01, eps, 0.0000001, r)
        err_known_var.append(np.linalg.norm(unk_var_mean-mean))

    return non_pr, err_unk_var, err_known_var, err_hada


def test_dim_local(mean0,sigma0,dims,eps,nPaths,nSamples,rfact=50,T=10,bGenerate=True,folder='',generate_func=generate_gauss_paths,cov_func=cov_id,prefix=''):
    err_nonpr = []
    err_unk_var = []
    err_known_var = []
    err_hada = []

    err_nonpr_paths = []
    err_unk_var_paths = []
    err_known_var_paths = []
    err_hada_paths = []

    for d in dims:
        print(d)
        mean = [mean0]*d
        cov = cov_func(sigma0,d)
        r = rfact*np.sqrt(d)
        u = 2.0*r
        assert(nSamples >= 3 * np.sqrt(d)*np.sqrt(T)/eps)
        non_pr, unk_var, known_var, hada = test_dist_local(mean,cov,eps,sigma0,nPaths,nSamples,d,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr,0.1))
        err_unk_var.append(trim_mean(unk_var,0.1))
        err_known_var.append(trim_mean(known_var,0.1))
        err_hada.append(trim_mean(hada,0.1))

        err_nonpr_paths.append(non_pr)
        err_unk_var_paths.append(err_nonpr)
        err_known_var_paths.append(err_known_var)
        err_hada_paths.append(err_hada)

    strdims = '|'.join([str(di) for di in dims])
    params = [['nPaths','nSamples','d','mean','sigma','eps','u','r'],[str(nPaths),str(nSamples),strdims,str(mean0),str(sigma0),str(eps),str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_d' + str(dims[0]) + '-' +str(dims[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(dims,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(dims,err_unk_var,strfolder,'err_unk_var.txt')
    write_output(dims,err_known_var,strfolder,'err_known_var.txt')
    write_output(dims,err_hada,strfolder,'err_hada.txt')

    write_output(dims,err_nonpr_paths,strfolder,'err_nonpr_paths.txt')
    write_output(dims,err_unk_var_paths,strfolder,'err_unk_var_paths.txt')
    write_output(dims,err_known_var_paths,strfolder,'err_known_var_paths.txt')
    write_output(dims,err_hada_paths,strfolder,'err_hada_paths.txt')

    return subfolder


def test_eps_local(mean0,sigma0,d,epss,nPaths,nSamples,rfact=50,bGenerate=True,folder='',generate_func=generate_gauss_paths,cov_func=cov_id,prefix=''):
    err_nonpr = []
    err_unk_var = []
    err_known_var = []
    err_hada = []

    err_nonpr_paths = []
    err_unk_var_paths = []
    err_known_var_paths = []
    err_hada_paths = []

    mean = [mean0]*d
    cov = cov_func(sigma0,d)
    r = rfact*np.sqrt(d)
    u = 2.0*r
    for eps in epss:
        print(eps)
        non_pr, unk_var, known_var, hada = test_dist_local(mean,cov,eps,sigma0,nPaths,nSamples,d,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr,0.1))
        err_unk_var.append(trim_mean(unk_var,0.1))
        err_known_var.append(trim_mean(known_var,0.1))
        err_hada.append(trim_mean(hada,0.1))

        err_nonpr_paths.append(non_pr)
        err_unk_var_paths.append(err_nonpr)
        err_known_var_paths.append(err_known_var)
        err_hada_paths.append(err_hada)

    strepss = '|'.join([str(eps) for eps in epss])
    params = [['nPaths','nSamples','d','mean','sigma','eps','u','r'],[str(nPaths),str(nSamples),str(d),str(mean0),str(sigma0),strepss,str(u),str(r)]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_d' + str(dims[0]) + '-' +str(dims[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(epss,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(epss,err_unk_var,strfolder,'err_unk_var.txt')
    write_output(epss,err_known_var,strfolder,'err_known_var.txt')
    write_output(epss,err_hada,strfolder,'err_hada.txt')

    write_output(epss,err_nonpr_paths,strfolder,'err_nonpr_paths.txt')
    write_output(epss,err_unk_var_paths,strfolder,'err_unk_var_paths.txt')
    write_output(epss,err_known_var_paths,strfolder,'err_known_var_paths.txt')
    write_output(epss,err_hada_paths,strfolder,'err_hada_paths.txt')

    return subfolder


def test_R_local(mean0,sigma0,d,eps,nPaths,nSamples,rfacts,bGenerate=True,folder='',generate_func=generate_gauss_paths,cov_func=cov_id,prefix=''):
    err_nonpr = []
    err_unk_var = []
    err_known_var = []
    err_hada = []

    err_nonpr_paths = []
    err_unk_var_paths = []
    err_known_var_paths = []
    err_hada_paths = []

    mean = [mean0]*d
    cov = cov_func(sigma0,d)
    Rs = [rfact*np.sqrt(d) for rfact in rfacts]
    for r in Rs:
        u = 2.0*r
        non_pr, unk_var, known_var, hada = test_dist_local(mean,cov,eps,sigma0,nPaths,nSamples,d,r,u,bGenerate=bGenerate,generate_func=generate_func,folder='')
        err_nonpr.append(trim_mean(non_pr,0.1))
        err_unk_var.append(trim_mean(unk_var,0.1))
        err_known_var.append(trim_mean(known_var,0.1))
        err_hada.append(trim_mean(hada,0.1))

        err_nonpr_paths.append(non_pr)
        err_unk_var_paths.append(err_nonpr)
        err_known_var_paths.append(err_known_var)
        err_hada_paths.append(err_hada)

    strRs = '|'.join([str(r) for r in Rs])
    params = [['nPaths','nSamples','d','mean','sigma','eps','u','r'],[str(nPaths),str(nSamples),str(d),str(mean0),str(sigma0),str(eps),str(u),strRs]]
    subfolder= str(nPaths) + prefix + 'paths_' + str(nSamples) + 'samples_R' + str(rfacts[0]) + '-' +str(rfacts[-1]) + '/'
    strfolder = folder + subfolder
    write_text(params,strfolder,'params.txt')
    write_output(Rs,err_nonpr,strfolder,'err_nonpr.txt')
    write_output(Rs,err_unk_var,strfolder,'err_unk_var.txt')
    write_output(Rs,err_known_var,strfolder,'err_known_var.txt')
    write_output(Rs,err_hada,strfolder,'err_hada.txt')

    write_output(Rs,err_nonpr_paths,strfolder,'err_nonpr_paths.txt')
    write_output(Rs,err_unk_var_paths,strfolder,'err_unk_var_paths.txt')
    write_output(Rs,err_known_var_paths,strfolder,'err_known_var_paths.txt')
    write_output(Rs,err_hada_paths,strfolder,'err_hada_paths.txt')

    return subfolder