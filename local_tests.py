from tests_functions import generate_gauss_paths, test_R_local, test_dim_local, test_eps_local
from utils import make_plot

dict_err = {}
dict_err['err_hada.txt'] = ('green','*','Shifted-CM')
dict_err['err_unk_var.txt'] = ('red','+',r'UnkVar')
dict_err['err_known_var.txt'] = ('magenta','+',r'KnownVar')
dict_err['err_nonpr.txt'] = ('blue','x','Non-private')
filelist = ['err_nonpr.txt','err_unk_var.txt','err_known_var.txt','err_hada.txt']

########################### Fig. 10: l2 error vs. d ########################### 
mean0 = [0.0, 5.0, 10.0]
sigma0 = 1.0
dims = [2, 4, 8, 16, 32]
nPaths = 50
nSamples = 10**5
folder = ['./results/test_local_d_sigma1_mean0_pp0.5/',
'./results/test_local_d_sigma1_mean5_pp0.5/',
'./results/test_local_d_sigma1_mean10_pp0.5/']
eps = 1.0
for i in range(len(mean0)):
    subfolder=test_dim_local(mean0[i],sigma0,dims,eps,nPaths,nSamples,rfact=20,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths)
    make_plot(folder[i]+subfolder,'test_local_d.pdf',r'$d$','l2 error',dict_names=dict_err,filenames=filelist,xscale='log',yscale='log',basex=2,basey=10)

# ########################### Fig. 11: l2 error vs. eps ########################### 
mean0 = [0.0, 5.0, 10.0]
sigma0 = 1.0
d = 8
nPaths = 50
nSamples = 10**5
folder = ['./results/test_local_d_sigma1_mean0_pp0.5/',
'./results/test_local_d_sigma1_mean5_pp0.5/',
'./results/test_local_d_sigma1_mean10_pp0.5/']
epss = [0.5, 1., 1.5, 2., 2.5]
for i in range(len(mean0)):
    subfolder=test_eps_local(mean0[i],sigma0,d,epss,nPaths,nSamples,rfact=20,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths)
    make_plot(folder[i]+subfolder,'test_local_eps.pdf',r'$\varepsilon$','l2 error',dict_names=dict_err,filenames=filelist,yscale='log',basey=10)

########################### Fig. 12: l2 error vs. R ########################### 
mean0 = [0.0, 5.0, 10.0]
sigma0 = 1.0
d = 8
nPaths = 50
nSamples = 10**5
folder = ['./results/test_local_R_sigma1_mean0_pp0.5/',
'./results/test_local_R_sigma1_mean5_pp0.5/',
'./results/test_local_R_sigma1_mean10_pp0.5/']
eps = 1.0
rfacts = [20., 65., 110., 155., 200.]
for i in range(len(mean0)):
    subfolder=test_R_local(mean0[i],sigma0,d,eps,nPaths,nSamples,rfacts=rfacts,bGenerate=True,folder=folder[i],generate_func=generate_gauss_paths)
    make_plot(folder[i]+subfolder,'test_local_R.pdf',r'$R$','l2 error',dict_names=dict_err,filenames=filelist,yscale='log',basey=10)