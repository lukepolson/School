import numpy as np
import pydicom
import matplotlib.pyplot as plt
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

algos = ['art', 'bart', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv']
iterations = [10, 250, 60, 60, 200, 200, 140, 200, 300, 360]
every_ns = [1, 10, 5, 5, 10, 10, 10, 10, 20, 30]

lam = 500
n = 400

A = imageio.imread('images/phantom4.png').sum(axis=2)
A_norm = functions.normalize_phantom(A)

Ts = np.array([np.asarray(imageio.imread(f'images/t{i}.png').sum(axis=2) > 0) for i in range(1,24)])
n = A.shape[0]
T = Ts.sum(axis=0).astype(bool)
T_masked = np.ma.masked_where(~T, T)

U = functions.get_tumour_dist(T, sigma=3)

dfile = np.load('tumour_dist_P4.npz')
PET_og, dpoints, dangles, prob_of_detections = dfile['PET_og'], dfile['dpoints'], dfile['dangles'], dfile['prob_of_detections']

PET_att, mask_att = functions.get_attenuated_PET(dpoints, dangles, prob_of_detections, n=n)

sino, rs, thetas =  functions.get_sinogram(dpoints, dangles)
sino_att, _, _ =  functions.get_sinogram(dpoints, dangles, mask=mask_att)

det_matrix = functions.compute_detection_matrix(A_norm, n=400, spacing=4)

info = {}
for algo, iteration, every_n in zip(algos, iterations, every_ns):
    print(algo)
    info_sub = {}
    n_iters, MSEs, im_best, iter_best  = functions.MSE_tomopy(sino_att, thetas, PET_og, T_masked,
                                    algo, num_iter=iteration, det_matrix=det_matrix, every_n=every_n)
    info[algo] = {'n_iter': n_iters, 'mse':MSEs, 'im_best': im_best, 'iter_best': iter_best}
    
df = pd.DataFrame(info)
df.to_csv('all_algos_P4.csv')