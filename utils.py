import os
import numpy as np
from urllib.request import urlretrieve
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_data(filename, num_images, dtype=np.float32):
    d = 28*28
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*d)
        data = np.frombuffer(buf, dtype=np.uint8).astype(dtype)
        data = (data / 255) #/ 28
        data = data.reshape(num_images, d)
        return data
     
def get_mnist_data(folder='data'):
    # if not os.path.exists('data'):
    #     os.mkdir('data')
    # filenames = ["train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
    # for name in filenames:
    #     urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)
    train_data = extract_data(folder+'/train-images-idx3-ubyte.gz', 60000)
    test_data = extract_data(folder+'/t10k-images-idx3-ubyte.gz', 10000)
    return train_data, test_data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def get_mnist_labels(folder='data'):
    # if not os.path.exists('data'):
    #     os.mkdir('data')
    # filenames = ["train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    # for name in filenames:
    #     if not os.path.exists("data/"+name):
    #         urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)
    train_labels = extract_labels(folder+'/train-labels-idx1-ubyte.gz', 60000)
    test_labels = extract_labels(folder+'/t10k-labels-idx1-ubyte.gz', 10000)
    return train_labels, test_labels

def write_output(x,y,folder,filename):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if (len(np.array(y).shape)) == 1:
        out = [x,y]
    else:
        out = []
        for i in range(len(x)):
            out.append([x[i]])
            out[i].extend(y[i])
    np.savetxt(folder+filename,np.transpose(out))

def write_text(x,folder,filename):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fOut = open(folder+filename,'w')
    for xi in x:
        fOut.write(','.join(xi)+'\n')
    fOut.close()

def save_paths(X,nPaths,nSamples,d,folder,filename):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    Y = np.reshape(X,(nPaths,d*nSamples))
    np.savetxt(folder+filename,np.transpose(Y))

def load_paths(filename,nPaths,nSamples,d):
    data = np.transpose(np.genfromtxt(filename))
    X = np.reshape(data,(nPaths,nSamples,d))
    return X

def make_plot(folder,savename,xlabel,ylabel,dict_names=None,filenames=[],xscale='linear',yscale='linear',basex=1,basey=1,loc='upper left',xlim=None,ylim=None):
    if not(dict_names==None):
        dict_data=dict_names
        filelist=filenames
    else:
        dict_data = {}
        dict_data['err_clip.txt'] = ('yellow','o','CM')
        dict_data['err_hada.txt'] = ('green','*','Shifted-CM')
        dict_data['err_t1.txt'] = ('darkviolet','+',r'COINPRESS $t=1$')
        dict_data['err_t2.txt'] = ('red','+',r'COINPRESS $t=2$')
        dict_data['err_t3.txt'] = ('magenta','+',r'COINPRESS $t=3$')
        dict_data['err_t4.txt'] = ('gray','+',r'COINPRESS $t=4$')
        dict_data['err_t10.txt'] = ('olive','+',r'COINPRESS $t=10$')
        dict_data['err_nonpr.txt'] = ('blue','x','Non-private')
        filelist = ['err_nonpr.txt','err_t1.txt','err_t2.txt','err_t3.txt','err_t4.txt','err_t10.txt','err_clip.txt','err_hada.txt']
    for filename in filelist:
        if not(filename in dict_data.keys()):
            continue
        data = np.transpose(np.genfromtxt(folder+filename))
        style = dict_data[filename]
        plt.plot(data[0],data[1],color=style[0],marker=style[1],label=style[2])
    plt.legend(loc=loc, framealpha=0.4)
    if not(xlim==None):
        plt.xlim(xlim)
    if not(ylim==None):
        plt.ylim(ylim)
    if not(xscale=='linear'):
        plt.xscale(xscale,base=basex)
    if not(yscale=='linear'):
        plt.yscale(yscale,base=basey)
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.grid()
    plt.savefig(folder+savename,bbox_inches='tight', pad_inches=0)
    plt.clf()
