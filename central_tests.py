
from tests_functions import cov_sigma, generate_gauss_paths, generate_mnist_paths, test_R, test_dim, test_dim_cov, test_rho, test_rho_emp, test_rho_emp_cov
from utils import make_plot

########################### Fig. 1: l2 error vs. d ########################### 
mean0 = [0.0, 5.0, 10.0]
sigma0 = 1.0
dims = [16,64,256,1024]
nPaths = 100
nSamples = 4000
folder = ['./results/test_d_sigma1_mean0_pp0.5/',
'./results/test_d_sigma1_mean5_pp0.5/',
'./results/test_d_sigma1_mean10_pp0.5/']
eps = 1.0
p = eps*eps*0.5
for i in range(len(mean0)):
    subfolder=test_dim(mean0[i],sigma0,dims,p,nPaths,nSamples,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths)
    make_plot(folder[i]+subfolder,'test_d.pdf',r'$d$','l2 error',xscale='log',yscale='log',basex=2,basey=10)

########################### Fig. 2: l2 error vs. rho ########################### 
mean0 = [0.0, 5.0, 10.0]
sigma0 = 1.0
d = 128
nPaths = 100
nSamples = 4000
folder = ['./results/test_rho_sigma1_mean0_d128/',
'./results/test_rho_sigma1_mean5_d128/',
'./results/test_rho_sigma1_mean10_d128/']
eps = [0.1, 0.3, 0.5, 0.7, 1.0]
ps = [ep*ep*0.5 for ep in eps]
for i in range(len(mean0)):
    subfolder=test_rho(mean0[i],sigma0,d,ps,nPaths,nSamples,bGenerate=True,rfact=50,folder=folder[i],generate_func=generate_gauss_paths)
    make_plot(folder[i]+subfolder,'test_rho.pdf',r'$\rho$','l2 error',yscale='log',basey=10,ylim=(0.1,100))


########################### Fig. 3: l2 error vs. d ########################### 
dict_err = {}
dict_err['err_hada.txt'] = ('green','*','Shifted-CM')
dict_err['err_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
dict_err['err_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
dict_err['err_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
dict_err['err_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
dict_err['err_t10.txt'] = ('olive','+',r'COINPRESS $t=10$')
dict_err['err_nonpr.txt'] = ('blue','x','Non-private')
filelist = ['err_nonpr.txt','err_t1.txt','err_t2.txt','err_t3.txt','err_t4.txt','err_t10.txt','err_hada.txt']
mean0 = 0.0
sigma0 = [10,100,1000]
dims = [16,64,256,1024]
nPaths = 100
nSamples = 4000
folder = ['./results/test_d_sigmak10_mean0_pp0.5/',
'./results/test_d_sigmak100_mean0_pp0.5/',
'./results/test_d_sigmak1000_mean0_pp0.5/']
eps = 1.0
p = eps*eps*0.5
for i in range(len(sigma0)):
    subfolder=test_dim(mean0,sigma0[i],dims,p,nPaths,nSamples,rfact=100,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths,cov_func=cov_sigma)
    make_plot(folder[i]+subfolder,'test_d.pdf',r'$d$','l2 error',dict_names=dict_err,filenames=filelist,xscale='log',yscale='log',basex=2,basey=10)

########################### Fig. 4, 5: l2 error vs. d ########################### 
dict_err = {}
dict_err['err_hada.txt'] = ('green','*','Shifted-CM')
dict_err['err_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
dict_err['err_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
dict_err['err_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
dict_err['err_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
dict_err['err_t5.txt'] = ('olive','+',r'COINPRESS $t=5$')
dict_err['err_nonpr.txt'] = ('blue','x','Non-private')
filelist = ['err_nonpr.txt','err_t1.txt','err_t2.txt','err_t3.txt','err_t4.txt','err_t5.txt','err_hada.txt']

dict_maha = {}
dict_maha['maha_hada.txt'] = ('green','*','Shifted-CM')
dict_maha['maha_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
dict_maha['maha_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
dict_maha['maha_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
dict_maha['maha_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
dict_maha['maha_t5.txt'] = ('olive','+',r'COINPRESS $t=5$')
dict_maha['maha_nonpr.txt'] = ('blue','x','Non-private')
filemaha = ['maha_nonpr.txt','maha_t1.txt','maha_t2.txt','maha_t3.txt','maha_t4.txt','maha_t5.txt','maha_hada.txt']

mean0 = 0.0
sigma0 = [10,100,1000]
dims = [16,64,256,1024]
nPaths = 100
nSamples = 4000
folder = ['./results/test_dcov_sigmak10_mean0_pp0.5/',
'./results/test_dcov_sigmak100_mean0_pp0.5/',
'./results/test_dcov_sigmak1000_mean0_pp0.5/']
eps = 1.0
p = eps*eps*0.5
for i in range(len(sigma0)):
    subfolder=test_dim_cov(mean0,sigma0[i],dims,p,nPaths,nSamples,rfact=100,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths,cov_func=cov_sigma)
    make_plot(folder[i]+subfolder,'test_d_err.pdf',r'$d$','l2 error',dict_names=dict_err,filenames=filelist,xscale='log',yscale='log',basex=2,basey=10)
    make_plot(folder[i]+subfolder,'test_d_maha.pdf',r'$d$','Mahalanobis error',dict_names=dict_maha,filenames=filemaha,xscale='log',yscale='log',basex=2,basey=10)

########################### Fig. 6: l2 error vs. R ########################### 
dict_err = {}
dict_err['err_hada.txt'] = ('green','*','Shifted-CM')
dict_err['err_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
dict_err['err_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
dict_err['err_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
dict_err['err_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
dict_err['err_t10.txt'] = ('olive','+',r'COINPRESS $t=10$')
dict_err['err_nonpr.txt'] = ('blue','x','Non-private')
filelist = ['err_nonpr.txt','err_t1.txt','err_t2.txt','err_t3.txt','err_t4.txt','err_t10.txt','err_hada.txt']
mean0 = [0.0, 5.0, 10.0]
sigma0 = 10
d = 128
nPaths = 100
nSamples = 4000
folder = ['./results/test_R_sigma10_mean0_d128/',
'./results/test_R_sigma10_mean5_d128/',
'./results/test_R_sigma10_mean10_d128/']
eps = 1.0
p = eps*eps*0.5
rfacts = [20.0, 65.0, 110.0, 155.0, 200.0]
for i in range(len(mean0)):
    subfolder=test_R(mean0[i],sigma0,d,p,nPaths,nSamples,rfacts=rfacts,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths,cov_func=cov_sigma)
    make_plot(folder[i]+subfolder,'test_R.pdf',r'$R$','l2 error',dict_names=dict_err,filenames=filelist,yscale='log',basey=10,ylim=(0.1,100))

########################### Fig. 7, MNIST: l2 error vs. rho ########################### 
dict_err = {}
dict_err['err_hada.txt'] = ('green','*','Shifted-CM')
dict_err['err_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
dict_err['err_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
dict_err['err_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
dict_err['err_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
dict_err['err_t10.txt'] = ('olive','+',r'COINPRESS $t=10$')
filelist = ['err_t1.txt','err_t2.txt','err_t3.txt','err_t4.txt','err_t10.txt','err_hada.txt']
digs= [0, 1, 2]
sigma0 = 1.0
d = 784
nPaths = 100
nSamples = 6000
folder = ['./results/test_mnist_dig0/',
'./results/test_mnist_dig1/',
'./results/test_mnist_dig2']
ps = [0.1, 0.2, 0.4, 0.8, 1.6]
for i in range(len(digs)):
    dig = digs[i]
    gen_mnist_func = lambda nS: generate_mnist_paths(dig,nS)
    subfolder=test_rho_emp(d,ps,nPaths,nSamples,rfact=50,folder=folder[i],generate_func=gen_mnist_func)
    make_plot(folder[i]+subfolder,'test_mnist_rho.pdf',r'$\rho$','l2 error',dict_names=dict_err,filenames=filelist,xscale='log',basex=2,yscale='log',basey=10)

########################### Fig. 8, MNIST: l2 error vs. rho ########################### 
dict_err = {}
dict_err['err_hada.txt'] = ('green','*','Shifted-CM')
dict_err['err_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
dict_err['err_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
dict_err['err_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
dict_err['err_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
dict_err['err_t5.txt'] = ('olive','+',r'COINPRESS $t=5$')
filelist = ['err_t1.txt','err_t2.txt','err_t3.txt','err_t4.txt','err_t5.txt','err_hada.txt']
digs= [0, 1, 2]
sigma0 = 1.0
d = 784
nPaths = 100
nSamples = 6000
folder = ['./results/test_mnist_cov_dig0/',
'./results/test_mnist_cov_dig1/',
'./results/test_mnist_cov_dig2']
ps = [0.1, 0.2, 0.4, 0.8, 1.6]
for i in range(len(digs)):
    dig = digs[i]
    gen_mnist_func = lambda nS: generate_mnist_paths(dig,nS)
    subfolder=test_rho_emp_cov(d,ps,nPaths,nSamples,rfact=50,folder=folder[i],generate_func=gen_mnist_func)
    make_plot(folder[i]+subfolder,'test_mnist_rho.pdf',r'$\rho$','l2 error',dict_names=dict_err,filenames=filelist,xscale='log',basex=2,yscale='log',basey=10)
