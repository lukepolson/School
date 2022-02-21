import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import sympy as smp
from scipy.stats import rv_continuous
plt.style.use(['science', 'notebook'])

'''PART 1: Auxillary Functions'''

# Read in all required data
df = pd.read_csv('water_data.txt', sep=' ', dtype=np.float32)
energy_d = df['Energy(MeV)'].values
coherent_d = df['Coherent(cm^2g^-1)'].values
compton_d = df['Compton(cm^2g^-1)'].values
photo_d = df['Photoelectric(cm^2g^-1)'].values
pairtrip_d = df['Pair+Triplet(cm^2g^-1)'].values
Etrans_d = df['Energy-Transfer(cm^2g^-1)'].values
Eabs_d = df['Energy-Absorption(cm^2g^-1)'].values
brem_d = df['1-g'].values
mu_d = coherent_d + compton_d+photo_d + pairtrip_d

# Interpolation functions for attenuations coefficients
mu_f = interp1d(energy_d, mu_d)
prob_coherent_f = interp1d(energy_d, coherent_d/mu_d)
prob_compton_f = interp1d(energy_d, compton_d/mu_d)
prob_photo_f = interp1d(energy_d, photo_d/mu_d)
prob_pairtrip_f = interp1d(energy_d, pairtrip_d/mu_d)    

def get_6MV_spectrum(N):
    cdf = np.array([2480,15000,27290,37590,46310,53760,60140,65680,70460,
                       74630,78290, 81510, 84330, 86860, 89090, 91060, 92790,
                       94330, 95670, 96840, 97850, 98710, 99420, 100000]) /100000
    energies = np.arange(0.25, 6.25, 0.25)
    return energies[np.searchsorted(cdf, np.random.rand(N))]

             
def get_interaction_type(E, Ecut=0.01):
    interactions = np.zeros(len(E), dtype=int)
    probs = np.array([prob_coherent_f(E),
                    prob_photo_f(E),
                    prob_compton_f(E),
                    prob_pairtrip_f(E)]).cumsum(axis=0)
    U = uniform(size=len(E))
    interactions[U<probs[0]] = 1
    interactions[(U>=probs[0])*(U<probs[1])] = 2
    interactions[(U>=probs[1])*(U<probs[2])] = 3
    interactions[(U>=probs[2])*(U<=probs[3])] = 4
    interactions[E<Ecut] = 0
    return interactions

class klein_gen(rv_continuous):
    def __init__(self, m=0.511):
        super(klein_gen, self).__init__()
        self.m = 0.511
    def get_pdf(self, x, E, norm=False):
        ratio = 1+E/self.m * (1-np.cos(x))
        pdf = np.sin(x) * (1/ratio)**2 * (ratio + 1/ratio - np.sin(x)**2)
        if norm:
            return pdf/(sum(pdf)*np.diff(x)[0])
        else:
            return pdf
    def get_rvs(self, E):
        N = len(E)
        finished = np.zeros(N).astype(bool)
        thetas = np.zeros(N)
        while True:
            still_need = ~finished ==True
            n = still_need.sum()
            x = uniform(size=n)*np.pi 
            y = uniform(size=n)*2
            updated_indices = y<self.get_pdf(x, E[still_need])
            updated_thetas = x[updated_indices]
            still_need[still_need] = updated_indices # a trick
            thetas[still_need] = np.copy(updated_thetas)
            finished[still_need] = True
            if finished.all():
                return thetas
    
klein = klein_gen()


the, phi, the_p, phi_p = smp.symbols(r"\theta \phi \theta' \phi'", positive=True)
u = smp.Matrix([smp.sin(the)*smp.cos(phi), smp.sin(the)*smp.sin(phi), smp.cos(the)])
v = smp.Matrix([smp.sin(the_p)*smp.cos(phi_p), smp.sin(the_p)*smp.sin(phi_p), smp.cos(the_p)])
zhat = u
yhat = smp.Matrix([-u[1], u[0], 0]); yhat = smp.trigsimp(yhat/yhat.norm())
xhat = smp.trigsimp(yhat.cross(zhat))
R = smp.Matrix([[xhat[0], yhat[0], zhat[0]],
               [xhat[1], yhat[1], zhat[1]],
               [xhat[2],yhat[2],zhat[2]]])
expr = smp.trigsimp(R@v).simplify()
expr = expr.subs(smp.Abs(smp.sin(the)), smp.sin(the)).simplify()
get_new_units = smp.lambdify([the,phi,the_p,phi_p], expr)

'''Convention: Theta is polar angle, alpha is azimuthal angle'''
def angle_rest_frame(theta_of_frame, phi_of_frame, theta_in_frame, phi_in_frame):
    n = np.squeeze(get_new_units(theta_of_frame, phi_of_frame, theta_in_frame, phi_in_frame), axis=1)
    # return phi, theta
    return np.arctan2(n[1],n[0]), np.arctan2(np.sqrt(n[0]**2+n[1]**2), n[2])
    

def compton_scatter(E, theta, m=0.511):
    photon = E / (1 + E/m * (1-np.cos(theta)))
    electron = E - photon     
    return photon, electron
    

# all energies in MeV
class RadSim:
    def __init__(self, E_p, X_p, Ang_p, E_bins, int_types, Ecut=0.01, Eshell = 543.1e-6, m=0.511,
                XYZ_lim=None):
        self.klein = klein_gen()
        self.int_types = int_types
        self.E_p = E_p
        self.X_p = X_p
        self.Ang_p = Ang_p
        self.Act_p = np.ones(len(E_p)).astype(bool)
        self.E_bins = E_bins
        self.IntType_p = np.zeros(len(E_p))
        self.Nint_p = np.zeros(len(E_p))
        self.IntHist_p = np.zeros((4,len(E_bins)-1)) #interaction histogram
        self.comptonratios = np.array([])
        self.comptonenergies = np.array([])
        self.Ecut = Ecut
        self.Eshell = Eshell
        self.m = m
        self.X_e = np.empty((3,0))
        self.E_e = np.array([])
        self.Ang_e = np.empty((2,0))
        self.XYZ_lim = XYZ_lim
    '''Update position of all photons'''
    def update_position(self):
        Phi0, Theta0 = self.Ang_p[:,self.Act_p]
        n0 = np.array([np.cos(Phi0)*np.sin(Theta0),
              np.sin(Phi0)*np.sin(Theta0),
              np.cos(Theta0)])
        L = -1/mu_f(self.E_p[self.Act_p]) *np.log(uniform(size=self.Act_p.sum()))
        self.X_p[:,self.Act_p] += L * n0 
    '''Get interaction type for all photons; update histograms accordingly'''
    def get_inttype(self):
        self.IntType_p[self.Act_p] = get_interaction_type(self.E_p[self.Act_p], Ecut=self.Ecut)
        # Update interaction type histograms
        for i in self.int_types:
            self.IntHist_p[i]+=np.histogram(self.E_p[(self.IntType_p==i)*(self.Act_p)], self.E_bins)[0]
    '''All 4 interactions change Energy and Angle Only, and give initial electron state'''
    def coherent(self, M):
        Phi0 = 2*np.pi*uniform(size=sum(M))
        Theta0 = np.arccos(1-2*uniform(size=sum(M)))
        # 1. Modify Direction
        self.Ang_p[:,M] = np.array((Phi0, Theta0))
        # 2. Modify Energy (no energy modifcation)
        # 3. Electron Initial conditions (none)
    def compton(self, M):
        # Get new angles
        Theta_new_p = self.klein.get_rvs(self.E_p[M]) 
        Phi_new_p = 2*np.pi*uniform(size=len(self.E_p[M]))
        # Get photon/electron energy from new angles
        E_p, E_e = compton_scatter(self.E_p[M], Theta_new_p)
        # Get new electron angle
        Theta_new_e = (1+E_p/self.m) * np.tan(Theta_new_p/2)
        Phi_new_e = Phi_new_p
        # Extra: Modify histograms for problem
        self.comptonenergies = np.append(self.E_p[M], self.comptonenergies)
        self.comptonratios = np.append(E_e/self.E_p[M], self.comptonratios)
        # 1. Modify Direction (rest frame)
        Phi0, Theta0 = angle_rest_frame(self.Ang_p[1][M], self.Ang_p[0][M], Theta_new_p, Phi_new_p)
        self.Ang_p[:,M] = np.array((Phi0, Theta0))
        # 2. Modify energy
        self.E_p[M] = E_p
        # 3. Electron initial conditions (also requires angular transform)
        self.X_e = np.append(self.X_e, self.X_p[:,M], axis=1)
        self.E_e = np.append(E_e, self.E_e)
        Phi_e, Theta_e = angle_rest_frame(self.Ang_p[1][M], self.Ang_p[0][M], -Theta_new_e, Phi_new_e)
        self.Ang_e = np.append(self.Ang_e, np.array((Phi_e, Theta_e)), axis=1)
    def photo(self, M):
        # 1. Modify Direction (no need, photon is lost)
        # 2. Modify Energy
        E_e = self.E_p[M] - self.Eshell
        self.E_p[M] = 0
        # 3. Electron initial conditions
        self.X_e = np.append(self.X_e, self.X_p[:,M], axis=1)
        self.E_e = np.append(E_e, self.E_e)
        self.Ang_e = np.append(self.Ang_e, self.Ang_p[:,M], axis=1) # assume no angle change
    def pair(self, M):
        # 1. Modify Direction (no need, photon lost)
        # 2. Modify Energy
        E_e = self.E_p[M] - 1.022
        self.E_p[M] = 0
        # 3. Electron initial conditions
        Theta_new_e = self.m/np.repeat(E_e,2) * ( 1 + 0.5*np.random.randn(2*len(E_e)))
        Phi_new_e = 2*np.pi*uniform(size=2*len(E_e))
        Phi_e, Theta_e = angle_rest_frame(np.repeat(self.Ang_p[1][M],2),
                                          np.repeat(self.Ang_p[0][M],2),
                                          Theta_new_e, Phi_new_e)
        self.X_e = np.append(self.X_e, np.repeat(self.X_p[:,M], 2, axis=1), axis=1)
        self.E_e = np.append(np.repeat(E_e,2)/2, self.E_e)
        self.Ang_e = np.append(self.Ang_e, np.array((Phi_e, Theta_e)), axis=1)
        
    '''For depositing energy of photons with less than 10keV'''
    def deposit(self, M):
        self.X_e = np.append(self.X_e, self.X_p[:,M], axis=1)
        self.E_e = np.append(self.E_e, self.E_p[M])
        self.Ang_e = np.append(self.Ang_e, np.array((np.zeros(sum(M)), np.zeros(sum(M)))), axis=1)
        
    '''Go through a single simulation iteration. Return True if finished''' 
    def iterate(self):
        if np.all(~self.Act_p):
            return True
        self.update_position()
        # If outside of box
        if self.XYZ_lim:
            self.Act_p = self.Act_p * (np.abs(self.X_p[0]) <= self.XYZ_lim[0]) \
                                    *(np.abs(self.X_p[1]) <= self.XYZ_lim[1]) \
                                    * (self.X_p[2] >= self.XYZ_lim[2]) \
                                    * (self.X_p[2] <= self.XYZ_lim[3])
        self.get_inttype()
        self.coherent((self.IntType_p==0)*self.Act_p)
        self.photo((self.IntType_p==1)*self.Act_p)
        self.compton((self.IntType_p==2)*self.Act_p)
        self.pair((self.IntType_p==3)*self.Act_p)
        self.deposit((self.E_p<self.Ecut)*self.Act_p)
        self.Act_p *= self.E_p >= self.Ecut
        return False
    '''Compute kerma and dose histograms in the region of interest'''
    def compute_volume_hists(self, binsx, binsy, binsz, dEdx=2, npoints=50, E_dose_cut=10e-3):
        # Get kerma histogram
        kerma_hist = np.histogramdd(self.X_e.T, [binsx, binsy, binsz],
                      weights=self.E_e)[0]
        # Get dose histogram
        dose_hist = np.zeros((len(binsx)-1, len(binsy)-1, len(binsz)-1))
        phi, theta = self.Ang_e[0], self.Ang_e[1]
        n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        for p in np.linspace(0,1,npoints):
            X = self.X_e + p*(1/dEdx)*(self.E_e-E_dose_cut) * n
            dose_hist += np.histogramdd(X.T, [binsx, binsy, binsz],
                              weights=self.E_e/npoints)[0]
        return kerma_hist, dose_hist