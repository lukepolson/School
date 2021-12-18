import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
from scipy import interpolate
from scipy import optimize
from scipy.signal import convolve
from skimage.filters import gaussian
plt.style.use(['science', 'notebook'])
import tomopy
import functions
import pandas as pd

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, rotate
from skimage.measure import profile_line

conv_shaper = np.array([[[-1,0,0],
                         [0,1,0],
                         [0,0,0]],
                        [[0,-1,0],
                          [0,1,0],
                          [0,0,0]],
                        [[0,0,-1],
                          [0,1,0],
                          [0,0,0]],
                        [[0,0,0],
                          [0,1,-1],
                          [0,0,0]],
                        [[0,0,0],
                          [0,1,0],
                          [0,0,-1]],
                        [[0,0,0],
                          [0,1,0],
                          [0,-1,0]],
                        [[0,0,0],
                          [0,1,0],
                          [-1,0,0]],
                        [[0,0,0],
                          [-1,1,0],
                          [0,0,0]],
                       ])

drk = np.array([np.sqrt(2),1]*4)

def dM(M):
    dM = [convolve(M, filt, 'same') for filt in conv_shaper]
    return np.array(dM)

def w(o,l,mu=0.1,nu=0.2,kap=1/3, M=500):
    dork = dM(o)
    dlrk = dM(l)
    mu = mu*M
    nu = nu*M
    kap = kap*l.max()
    return np.exp(-(dork/mu)**2) + \
           (1-np.exp(-(dork/mu)**2))*np.exp(-(dork/nu)**2)*np.exp(-(dlrk/kap)**2)

def dwdo(o,l,mu=0.1,nu=0.2,kap=1/3, M=500):
    mu = mu*M
    nu = nu*M
    kap = kap*l.max()
    dork = dM(o)
    dlrk = dM(l)
    fac = 2/(mu**2 * nu**2)
    t1 = np.exp(-(dork/mu)**2)
    t2 = np.exp(-(dork/nu)**2)
    t3 = np.exp(-(dlrk/kap)**2)
    return fac * (mu**2 *t1*t2*t3 - mu**2 * t3*t2 + nu**2 * t1*t2*t3 - nu**2 * t1) * dork
                    
def f(o, i, l, h, lam):
    o_2D = o.reshape(400,400).copy()
    term1 = np.sum((i-convolve(o_2D,h, 'same'))**2) 
    term2 = np.sum(w(o_2D,l)/drk[:,np.newaxis,np.newaxis]*dM(o_2D)**2)
    return (term1 + lam*term2)
def gradf(o, i, l, h, lam):
    o_2D = o.reshape(400,400).copy()
    term1 = 2*convolve((convolve(o_2D,h, 'same')-i), h, 'same')
    term2 = 4*np.sum(w(o_2D,l)/drk[:,np.newaxis,np.newaxis]*dM(o_2D), axis=0)
    term3 = 2*np.sum(1/drk[:,np.newaxis,np.newaxis] * dwdo(o_2D,l)*dM(o_2D)**2, axis=0)
    grad = term1 + lam*(term2+term3) 
    return grad.ravel()