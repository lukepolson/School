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
import importlib
import paper_f_old as pfo 
import paper_f_newnew as pfn 

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, rotate
from skimage.measure import profile_line

def get_gaus_kernel(n, sig):
    xv, yv = np.meshgrid(np.arange(n), np.arange(n))
    xv = xv - xv.max()/2
    yv = yv -  yv.max()/2
    gaus_kernel = 1/np.sqrt(2*np.pi*sig**2) * np.exp(-(xv**2+yv**2)/(2*sig**2))
    return gaus_kernel/gaus_kernel.sum()

lam = 500
n = 400

A1 = imageio.imread('images/phantom1.png').sum(axis=2)
A2 = imageio.imread('images/phantom2.png').sum(axis=2)
A3 = imageio.imread('images/phantom3.png').sum(axis=2)
A4 = imageio.imread('images/phantom4.png').sum(axis=2)
A1_norm = functions.normalize_phantom(A1)
A2_norm = functions.normalize_phantom(A2)
A3_norm = functions.normalize_phantom(A3)
A4_norm = functions.normalize_phantom(A4)

A = A1
A_norm = A1_norm

Ts = np.array([np.asarray(imageio.imread(f'images/t{i}.png').sum(axis=2) > 0) for i in range(1,24)])
n = A1.shape[0]
T = Ts.sum(axis=0).astype(bool)
T_masked = np.ma.masked_where(~T, T)

U = functions.get_tumour_dist(T, sigma=3)

dfile = np.load('tumour_dist_P1.npz')
PET_og, dpoints, dangles, prob_of_detections = dfile['PET_og'], dfile['dpoints'], dfile['dangles'], dfile['prob_of_detections']

PET_att, mask_att = functions.get_attenuated_PET(dpoints, dangles, prob_of_detections, n=n)

dpoints_psf = np.zeros((500,2)) + 200
dangles_psf = np.linspace(0, 2*np.pi, 500)

sino, rs, thetas =  functions.get_sinogram(dpoints, dangles)
sino_att, _, _ =  functions.get_sinogram(dpoints, dangles, mask=mask_att)
sino_psf, rs_psf, thetas_psf = functions.get_sinogram(dpoints_psf, dangles_psf)

for algo in ['art', 'bart', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv']:

    img = np.load('images/algo_images_P1/{}.npz'.format(algo))['img']

    i = gaussian(img, sigma=2)
    l = T.astype(float)
    h = get_gaus_kernel(9, 2.5)
    o = i.copy().ravel()

    lam = 1
    lam2 = 3

    res = optimize.minimize(pfn.f,o,args=(i,l,h,lam, lam2), jac=pfn.gradf, method='CG',
                           options=dict(maxiter=100, gtol=160000*1e-5))
    np.savez("images/algo_images_P1/PVC/{}.npz".format(algo), img=res.x.reshape(400,400))