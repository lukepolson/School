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

correction = 'nlm'
os_num_blocks = 3

algos = ['art', 'bart', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv']
iterations = [10, 250, 60, 30, 40, 40, 140, 200, 300, 360]
every_ns = [1, 10, 5, 3, 4, 4, 10, 10, 20, 30] # change osem to 5

lam = 500
n = 400
L_val = 1/2
B_val = 1/5

A = imageio.imread('images/phantom1.png').sum(axis=2)
A_norm = functions.normalize_phantom(A)

L = np.asarray(imageio.imread(f'images/liv.png').sum(axis=2))>260
B = np.asarray(imageio.imread(f'images/background.png').sum(axis=2))>260
Ts = np.array([np.asarray(imageio.imread(f'images/t{i}.png').sum(axis=2) > 0) for i in range(1,24)])
Ts_non_overlap = []
for T in Ts:
    if not np.any(T*L):
        Ts_non_overlap.append(T)
T = np.array(Ts_non_overlap).sum(axis=0).astype(bool)

# Further adjustments
B = B^(T+L) #background locations

T_masked = np.ma.masked_where(~T, T)
L_masked = np.ma.masked_where(~L, L)
B_masked = np.ma.masked_where(~B, B)
masks = [T_masked, L_masked, B_masked]
mask_names = ['T', 'L', 'B']

U = functions.get_tumour_dist(T, sigma=3)
U[L] += L_val
U[B] += B_val

dfile = np.load('tumour_dist_P1_livback.npz')
PET_og, dpoints, dangles, prob_of_detections = dfile['PET_og'], dfile['dpoints'], dfile['dangles'], dfile['prob_of_detections']

PET_att, mask_att = functions.get_attenuated_PET(dpoints, dangles, prob_of_detections, n=n)

sino, rs, thetas =  functions.get_sinogram(dpoints, dangles)
sino_att, _, _ =  functions.get_sinogram(dpoints, dangles, mask=mask_att)
sino_prob = functions.estimate_prob_of_detections_rtheta(rs, thetas, A_norm, num=1000)

info = {}
for algo, iteration, every_n in zip(algos, iterations, every_ns):
    print(algo)
    
    extra_args = {}
    if 'os' in algo: # Number of subsets in ordered-subset expectation maximum
        extra_args['num_block'] = os_num_blocks

    n_iters, data, im_bests, iter_bests  = functions.MSE_tomopy_multicorr_multimask(sino_att/sino_prob, thetas, PET_og, masks, mask_names, correction, algo, num_iter=iteration, every_n=every_n, **extra_args)
    info[algo] = {'n_iter': n_iters, 'data':data, 'im_bests': im_bests, 'iter_bests': iter_bests}

if correction is None:
    name = ''
else:
    name = correction+'_'

df = pd.DataFrame(info)
df.to_pickle(f'all_algos_P1_{name}livback.pkl')