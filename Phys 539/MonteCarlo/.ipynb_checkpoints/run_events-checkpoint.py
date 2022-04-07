'''
For running seperate parallel jobs on Compute Canada
'''

import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import sympy as smp
from scipy.stats import rv_continuous
plt.style.use(['science', 'notebook'])
from functions import *
import functions as f
import matplotlib
import scipy.ndimage as ndimage
from skimage.transform import rotate
import sys

i = int(sys.argv[1])
path = '/home/lpolson/projects/def-mlefebvr/lpolson/Phys539_data/'

# Specify number of initial photons and voxel size in cm^3
N = 10000000
voxel = 0.1
binsx = bins = np.arange(-15-voxel/2,15+3*voxel/2,voxel)
binsy = bins = np.arange(-15-voxel/2,15+3*voxel/2,voxel)
binsz = np.arange(0,50+voxel,voxel)

# Get initial conditions of photons
E = get_6MV_spectrum(N)
R = np.sqrt(2) * np.sqrt(np.log(6/E))
phi = 2*np.pi*np.random.uniform(size=N)
x = R*np.cos(phi)+(1e-19) * np.random.randn(len(R))
y = R*np.sin(phi)+(1e-19) * np.random.randn(len(R))
z=np.zeros(N)
X = np.array([x, y, z])
theta = np.arctan(R/100)
Ang = np.array([phi, theta])
Ebins = 10.0**(np.linspace(-3,1,100))
int_types = np.arange(4)
r = RadSim(E, X, Ang, Ebins, int_types, binsx, binsy, binsz, XYZ_lim=[15.,15.,0.,50.])

# Run simulation
while True:
    if r.iterate():
        break

# Get histograms
prim_hist, tot_hist, kerma_hist, dose_hist = r.compute_volume_hists(dEdx=2)

# Save histograms
np.save(path+'prim_hist_{}'.format(i), prim_hist)
np.save(path+'tot_hist_{}'.format(i), tot_hist)
np.save(path+'kerma_hist_{}'.format(i), kerma_hist)
np.save(path+'dose_hist_{}'.format(i), dose_hist)