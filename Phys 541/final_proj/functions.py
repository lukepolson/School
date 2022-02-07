import numpy as np
import pydicom
import matplotlib.pyplot as plt
import imageio
from scipy import interpolate
from skimage.filters import gaussian
plt.style.use(['science', 'notebook'])
import tomopy

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, rotate
from skimage.measure import profile_line
from skimage.restoration import denoise_nl_means, estimate_sigma, richardson_lucy, unsupervised_wiener, wiener

'''Initial'''

def normalize_phantom(A, p=0.5):
    n = A.shape[0]
    mu = -np.log(p)/n
    m = np.mean(A[A>0])
    return A*mu/m

def get_tumour_dist(T, sigma=3):
    U = gaussian(T, sigma=3)
    U[~T]=0
    return U

def get_original_PET(U, T, lam=300, seed=0):
    np.random.seed(seed)
    decay_values = np.random.poisson(lam=lam*U, size=U.shape)
    decay_values = decay_values[T]
    original = np.zeros(T.shape)
    original[T] = decay_values
    np.random.seed(seed)
    decay_angles = np.random.uniform(size=decay_values.sum())*2*np.pi
    decay_points = np.repeat(np.argwhere(T), decay_values, axis=0)
    return original, decay_points, decay_angles

def compute_prob_of_detections(A, decay_points, decay_angles, n=400, num=1000):
    diameter = np.sqrt(2*n**2)
    prob_of_detections = []
    for i, (dp, da) in enumerate(zip(decay_points, decay_angles)):
        # Get initial and final points
        x1 = dp[0] - np.cos(da)* diameter
        x2 =  dp[0] + np.cos(da)* diameter
        y1 = dp[1] - np.sin(da)* diameter
        y2 =  dp[1] + np.sin(da)* diameter
        xx = np.linspace(x1, x2, num)
        yy = np.linspace(y1, y2, num)
        mask = (xx>0) * (xx<n) * (yy>0) * (yy<n)
        xx = xx[mask].astype(int)
        yy = yy[mask].astype(int)
        dL = 2*diameter / num
        prob_of_detections.append(np.exp(-A[xx,yy].sum()*dL))
    return np.array(prob_of_detections)

def get_attenuated_PET(decay_points, decay_angles, prob_of_detections, n=200):
    xs, ys = decay_points.T
    mask = prob_of_detections > np.random.uniform(size=len(prob_of_detections))
    attenuated,_,_,_=plt.hist2d(xs[mask], ys[mask], bins=[np.arange(-0.5,n+0.5),np.arange(-0.5,n+0.5)])
    plt.close()
    return attenuated, mask

def get_sinogram(decay_points, decay_angles, mask=[False], n=400, n_theta_bins=61):
    center = (n//2, n//2)
    xs, ys = decay_points.T
    if any(mask):
        xs = xs[mask]
        ys = ys[mask]
        decay_angles = decay_angles[mask]
    
    rs = (xs - center[0]) * np.cos(decay_angles+np.pi/2) + (ys - center[1]) * np.sin(decay_angles+np.pi/2)
    thetas = decay_angles
    rs_bins = np.arange(-n//2-0.5, n//2+0.5, 1)
    theta_bins = np.linspace(0,2*np.pi,n_theta_bins)
    sino,_,_,_ = plt.hist2d(rs,thetas, bins=[rs_bins, theta_bins])
    plt.close()
    return sino, rs_bins, theta_bins

'''Reconstruction'''

def estimate_prob_of_detections(x, y, A, thetas = np.linspace(0,np.pi,40), num=1000):
    n = A.shape[0]
    diameter = np.sqrt(2*n**2)
    prob_of_detections = []
    for theta in thetas:
        # Get initial and final points
        x1 = x - np.cos(theta)* diameter
        x2 =  x + np.cos(theta)* diameter
        y1 = y - np.sin(theta)* diameter
        y2 =  y + np.sin(theta)* diameter
        xx = np.linspace(x1, x2, num)
        yy = np.linspace(y1, y2, num)
        mask = (xx>0) * (xx<n) * (yy>0) * (yy<n)
        xx = xx[mask].astype(int)
        yy = yy[mask].astype(int)
        dL = 2*diameter / num
        prob_of_detections.append(np.exp(-A[xx,yy].sum()*dL))
    return np.mean(prob_of_detections)

def estimate_prob_of_detections_rtheta(rs, thetas, A, num=1000):
    n = A.shape[0]
    diameter = np.sqrt(2*n**2)
    # from histogram edges to centers
    thetas = thetas[:-1]+np.diff(thetas)[0]/2
    rs = rs[:-1]+0.5
    print(len(rs))
    prob_of_detections = np.zeros([len(rs),len(thetas)])
    for i,r in enumerate(rs):
        for j,theta in enumerate(thetas):
            x = r*np.cos(theta+np.pi/2) + n//2
            y = r*np.sin(theta+np.pi/2) + n//2
            x1 = x - np.cos(theta)* diameter
            x2 =  x + np.cos(theta)* diameter
            y1 = y - np.sin(theta)* diameter
            y2 =  y + np.sin(theta)* diameter
            xx = np.linspace(x1, x2, num)
            yy = np.linspace(y1, y2, num)
            mask = (xx>0) * (xx<n) * (yy>0) * (yy<n)
            xx = xx[mask].astype(int)
            yy = yy[mask].astype(int)
            dL = 2*diameter / num
            prob_of_detections[i][j] = np.exp(-A[xx,yy].sum()*dL)
    return prob_of_detections

def compute_detection_matrix(A, n=400, spacing=4):
    s_array = np.arange(0,n,spacing)
    xv, yv = np.meshgrid(s_array, s_array)
    ps = np.vectorize(estimate_prob_of_detections, excluded=['A', 'thetas'])(xv, yv, A=A)
    ps_inter = RectBivariateSpline(s_array, s_array, ps)
    xv, yv = np.meshgrid(np.arange(0,n,1), np.arange(0,n,1))
    return np.vectorize(ps_inter)(xv,yv)

def compute_recon_tomopy(sino, thetas, algo_name, **kwargs):
    # Deal with any extra arguments passed to filtering:
    extra_args = {}
    for name in kwargs:
        if kwargs[name] is not None:
            extra_args[name] = kwargs[name]
            
    proj = np.expand_dims(sino.T, axis=1)
    thetas = thetas[:-1]+np.diff(thetas)[0]/2
    im = tomopy.recon(proj, thetas, algorithm=algo_name, **extra_args)
    return im[0]*len(thetas)

def MSE_tomopy(sino, thetas, original, T, algo_name, num_iter, det_matrix=None, every_n=1, **kwargs):
    # Deal with any extra arguments passed to filtering:
    extra_args = {}
    for name in kwargs:
        if kwargs[name] is not None:
            extra_args[name] = kwargs[name]
            
    MSEs = []
    VARs = []
    BIASs = []
    n_iters = []
    MSE_best = np.inf
    for i in range(1,num_iter+2):
        if (i-1)%every_n==0:
            im = compute_recon_tomopy(sino, thetas, algo_name, num_iter=i, **extra_args);
            if det_matrix is not None:
                im = im/det_matrix
            MSE = np.mean((original[T]-im[T])**2)
            VAR = np.var(original[T]-im[T])
            BIAS = np.mean(original[T]-im[T])
            if MSE<MSE_best:
                MSE_best = MSE
                iter_best = i
                im_best = im
            MSEs.append(MSE)
            VARs.append(VAR)
            BIASs.append(BIAS)
            n_iters.append(i)
    return np.array(n_iters), np.array(MSEs), np.array(VARs), np.array(BIASs), im_best, iter_best

def MSE_tomopy_multimask(sino, thetas, original, masks, mask_names, algo_name, num_iter, every_n=1, **kwargs):
    # Deal with any extra arguments passed to filtering:
    extra_args = {}
    for name in kwargs:
        if kwargs[name]:
            extra_args[name] = kwargs[name]
    
    data = {}
    for name in mask_names:
                data[name] = {'MSE': [], 'VAR': [], 'BIAS':[]}
    n_iters = []
    MSE_bests = [np.inf]*len(masks)
    iter_bests = [0]*len(masks)
    im_bests = [0]*len(masks)
    for i in range(1,num_iter+2):
        if (i-1)%every_n==0:
            im = compute_recon_tomopy(sino, thetas, algo_name, num_iter=i, **extra_args);
            for j, (M, name) in enumerate(zip(masks, mask_names)):
                MSE = np.mean((original[M]-im[M])**2)
                VAR = np.var(original[M]-im[M])
                BIAS = np.mean(original[M]-im[M])
                if MSE<MSE_bests[j]:
                    MSE_bests[j] = MSE
                    iter_bests[j] = i
                    im_bests[j] = im
                data[name]['MSE'].append(MSE)
                data[name]['VAR'].append(VAR)
                data[name]['BIAS'].append(BIAS)
            n_iters.append(i)
    return np.array(n_iters), data, im_bests, iter_bests

def MSE_tomopy_multicorr_multimask(sino, thetas, original, masks, mask_names, correction, algo_name, num_iter, every_n=1, **kwargs):
    # Deal with any extra arguments passed to filtering:
    extra_args = {}
    for name in kwargs:
        if kwargs[name]:
            extra_args[name] = kwargs[name]
    
    data = {}
    for name in mask_names:
                data[name] = {'MSE': [], 'VAR': [], 'BIAS':[]}
    n_iters = []
    MSE_bests = [np.inf]*len(masks)
    iter_bests = [0]*len(masks)
    im_bests = [0]*len(masks)
    for i in range(1,num_iter+2):
        if (i-1)%every_n==0:
            im = compute_recon_tomopy(sino, thetas, algo_name, num_iter=i, **extra_args);
            if correction=='wiener':
                im = wiener(im, get_gaus_kernel(9, 0.5), balance=14.5, clip=False)
            elif correction=='nlm':
                print(correction)
                im = denoise_nl_means(im, patch_size=10, patch_distance=30, h=30)
            for j, (M, name) in enumerate(zip(masks, mask_names)):
                MSE = np.mean((original[M]-im[M])**2)
                VAR = np.var(original[M]-im[M])
                BIAS = np.mean(original[M]-im[M])
                if MSE<MSE_bests[j]:
                    MSE_bests[j] = MSE
                    iter_bests[j] = i
                    im_bests[j] = im
                data[name]['MSE'].append(MSE)
                data[name]['VAR'].append(VAR)
                data[name]['BIAS'].append(BIAS)
            n_iters.append(i)
    return np.array(n_iters), data, im_bests, iter_bests
        