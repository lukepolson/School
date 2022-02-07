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

def w(l, kap=1/3):
    dlrk = dM(l)
    return np.exp(-(dlrk/kap)**2)
              
def f(o, i, l, h, lam, lam2, rho=5):
    o_2D = o.reshape(400,400).copy()
    term1 = np.sum((i-convolve(o_2D,h, 'same'))**2) 
    term2 = np.sum(w(l)/drk[:,np.newaxis,np.newaxis]*dM(o_2D)**2)
    term3 = np.sum(np.exp(-rho*l) * o_2D**2)
    return term1 + lam*term2 + lam2*term3

def gradf(o, i, l, h, lam, lam2, rho=5):
    o_2D = o.reshape(400,400).copy()
    term1 = 2*convolve((convolve(o_2D,h, 'same')-i), h, 'same')
    term2 = 4*np.sum(w(l)/drk[:,np.newaxis,np.newaxis]*dM(o_2D), axis=0)
    term3 = 2*np.exp(-rho*l)*o_2D
    grad = term1 + lam*term2  + lam2*term3
    return grad.ravel()