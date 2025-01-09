import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import random
import nbformat
import os

#############################################################################
#############################################################################
#############################################################################
#############################################################################

## POINT SPREAD FUNCTION (GAUSSIAN KERNEL)
def gaussian(domain, sigma):
    '''Gaussian centered in zero'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))

def squared_gaussian(domain, sigma):
    '''Squared Gaussian centered in zero'''
    return np.power(gaussian(domain, sigma),2)

# def fourier_measurements(x, fc):
#     ii = np.complex(0,1)
#     x = np.array(x)
#     fc_vect = np.arange(-fc, fc+1)
#     result = np.exp(-2* ii * np.pi * (fc_vect[:,None] @ x[:,None].T))
#     return result

#############################################################################
#############################################################################
#############################################################################
#############################################################################

## MEASURE 1D
class Measure:
    def __init__(self, amplitude, position, typeDomain='segment'):
        if len(amplitude) != len(position):
            raise ValueError('Amplitude and position have not the same number of entries!')
        self.a = amplitude
        self.x = position
        self.N = len(amplitude)
        self.Xtype = typeDomain


    def __add__(self, m):
        return Measure(self.a + m.a, self.x + m.x)


    def __eq__(self, m):
        if m == 0 and (self.a == [] and self.x == []):
            return True
        elif isinstance(m, self.__class__):
            return self.__dict__ == m.__dict__
        else:
            return False


    def kernel(self, X, sigma, ker='gaussian'):
        '''Kernel applied to the discrete measure, i.e. convolution with PSF'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X*0
        if ker == 'gaussian':
            for i in range(0,N):
                acquis += a[i] * gaussian(X - x[i], sigma)
            return acquis
        elif ker == 'squared_gaussian':
            for i in range(0,N):
                acquis += a[i] * squared_gaussian(X - x[i], sigma)
            return acquis
#         elif ker == 'fourier':
#             a = np.array(a)
#             acquis = fourier_measurements(x, F_C) @ a
#             return acquis
        else:
            raise TypeError("Unknown kernel")


    def acquisition(self, nv, bg, X, sigma, ker='gaussian'):
        '''Forward operator + Poisson noise '''
        acquis = self.kernel(X, sigma, ker=ker)+bg
        if nv == 0:
            return acquis
#         if ker == 'fourier':
#             w = np.fft.fft(np.random.randn(acquis.shape[0]))
#             w = np.fft.fftshift(w)
#             w /= np.linalg.norm(w)
#             acquis += nv * w
#             return acquis
        if ker == 'gaussian':
            acquis = np.random.poisson(acquis * nv) / nv
            return acquis
        else:
            raise TypeError("Unknown kernel")


    def tv(self):
        '''TV norm of the measure'''
        return np.linalg.norm(self.a, 1)

    def prune(self, tol=1e-4):
        '''Remove diracs with zero amplitude'''
        # nnz = np.count_nonzero(self.a)
        nnz_a = np.array(self.a)
        nnz = np.abs(nnz_a) > tol
        nnz_a = nnz_a[nnz]
        nnz_x = np.array(self.x)
        nnz_x = nnz_x[nnz]
        m = Measure(nnz_a.tolist(), nnz_x.tolist())
        return m

#############################################################################
#############################################################################
#############################################################################
#############################################################################
    
## FORWARD OPERATOR 
def phi(m, domain, sigma, ker='gaussian'):
    return m.kernel(domain, sigma, ker=ker)


def phi_vector(a, x, domain, sigma, ker='gaussian'):
    m_tmp = Measure(a, x)
    return(m_tmp.kernel(domain, sigma, ker=ker))

def phiAdjoint(acquis, domain, sigma, ker='gaussian'):
    if ker == 'gaussian':
        #return np.convolve(gaussian(X_big),acquis,'valid')/N_ech
        PSF = gaussian(domain-0.5, sigma)
        PSF = PSF/np.max(PSF)
        return np.real(np.fft.ifft(np.fft.fft(np.fft.fftshift(PSF))*np.fft.fft(acquis)))
#     if ker == 'fourier':
#         cont_fct = fourier_measurements(domain, F_C).T @ acquis
#         return np.flip(np.real(cont_fct))
    else:
        raise TypeError
        
def phi_amplitude_adjoint(vect, pos, domain, sigma, ker='gaussian'): 
    N = np.size(pos)
    adj = np.zeros([N,])
    for i in np.arange(N):
        adj[i] = np.sum(gaussian(domain-pos[i], sigma)*vect)
    return adj

#############################################################################
#############################################################################
#############################################################################
#############################################################################

## SIMULATION OF RANDOM MOLECULES 

def rand_gt(N_mol):
    m = Measure([],[])  
    for l in np.arange(N_mol):
        x = random.random()
        a = random.uniform(0.6,1.4)        
        m1 = Measure([a],[x])
        m = m+m1
    return m 

#############################################################################
#############################################################################
#############################################################################
#############################################################################

def L2TV_cost_funct(m, X, acquis, bg, par_reg, sigma, ker='gaussian'):
    aus = m.kernel(X, sigma, ker=ker) + bg
    fidelity = 0.5*np.linalg.norm(acquis - aus)**2
    penalty = par_reg*m.tv()
    return(fidelity + penalty)

def etak_L2(measure, X, acquis, bg, par_reg, sigma, ker='gaussian'):
    if ker == 'gaussian':
        eta = 1/par_reg*phiAdjoint(acquis - phi(measure, X, sigma, ker=ker)-bg,
                                 X, sigma, ker=ker)
        return eta
#     if ker == 'fourier':
#         eta = 1/par_reg*phiAdjoint(acquis - phi(measure, X, sigma, ker=ker)-bg,
#                                  X, sigma, ker=ker)
#         return eta

#############################################################################
#############################################################################
#############################################################################
#############################################################################

def KLTV_cost_funct(m, X, acquis, bg, par_reg, sigma, ker='gaussian'):
    aus = m.kernel(X, sigma, ker=ker)
    fidelity = np.sum(aus+bg-acquis+acquis*np.log(acquis) - acquis*np.log(aus+bg))
    penalty = par_reg*m.tv()
    return(fidelity + penalty)


def etak_KL(measure, X, acquis, bg, par_reg, sigma, ker='gaussian'):
    if ker == 'gaussian':
        eta = np.ones(len(X))-np.divide(acquis,phi(measure, X, sigma, ker=ker)+bg)
        eta = -1/par_reg*phiAdjoint(eta, X, sigma, ker=ker)
        return eta
#     if ker == 'fourier':
#         eta = 1/par_reg*phiAdjoint(acquis - phi(measure, X, sigma, ker=ker),
#                                  X, sigma, ker=ker)
        return eta

#############################################################################
#############################################################################
#############################################################################
#############################################################################

def plot_results(rec_m, nrj, par_reg, X, y, certificat_V=0, gt=0, L2=True):
#     if gt!=0:
#         try:
#             wasser = wasserstein_distance(rec_m.x, gt.x, rec_m.a, gt.a)
#             print(f'2-Wasserstein distance : {wasser}')
#         except ValueError:
#             print("[!] Flat metric is currently only implemented for " +
#                   "non-negative spikes") 

    if rec_m != 0 and np.isscalar(par_reg):
        plt.figure(figsize=(24,6))
        plt.subplot(131)
        plt.plot(X,y, label='$y$', linewidth=1.7)
        plt.stem(rec_m.x, rec_m.a, label='$m_{a,x}$', linefmt='C1--', 
              markerfmt='C1o', use_line_collection=True, basefmt=" ")
        if gt!=0:
            plt.stem(gt.x, gt.a, label='$m_{a_0,x_0}$', linefmt='C2--', 
              markerfmt='C2o', use_line_collection=True, basefmt=" ")
            # plt.xlabel('$x$', fontsize=18)
            plt.title(' Rec $m_{a,x}$ VS ground truth $m_{a_0,x_0}$', fontsize=20)
        else: 
            plt.title(' Rec $m_{a,x}$ (ground truth not known)', fontsize=20)
        plt.grid()
        plt.legend()

        plt.subplot(132)
        plt.plot(X, certificat_V, 'r', linewidth=2)
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2.5)
        if L2:
            plt.axhline(y=-1, color='gray', linestyle='--', linewidth=2.5)
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Amplitude with $\lambda=${par_reg:.1e}', fontsize=18)
        plt.title('Certificat $\eta_V$ of $m_{a,x}$', fontsize=20)
        plt.grid()

        plt.subplot(133)
        plt.plot(nrj, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('Decreasing energy', fontsize=20)
        plt.grid() 

    elif rec_m != 0:
        plt.figure(figsize=(24,6))
        plt.subplot(141)
        plt.plot(X,y, label='$y$', linewidth=1.7)
        plt.stem(rec_m.x, rec_m.a, label='$m_{a,x}$', linefmt='C1--', 
              markerfmt='C1o', use_line_collection=True, basefmt=" ")
        if gt!=0:
            plt.stem(gt.x, gt.a, label='$m_{a_0,x_0}$', linefmt='C2--', 
              markerfmt='C2o', use_line_collection=True, basefmt=" ")
            # plt.xlabel('$x$', fontsize=18)
            plt.title(' Rec $m_{a,x}$ VS ground truth $m_{a_0,x_0}$', fontsize=20)
        else: 
            plt.title(' Rec $m_{a,x}$ (ground truth not known)', fontsize=20)
        plt.grid()
        plt.legend()

        plt.subplot(142)
        plt.plot(X, certificat_V, 'r', linewidth=2)
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2.5)
        if L2:
            plt.axhline(y=-1, color='gray', linestyle='--', linewidth=2.5)
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Amplitude with $\lambda=${par_reg[len(par_reg)-1]:.1e}', fontsize=18)
        plt.title('Certificat $\eta_V$ of $m_{a,x}$', fontsize=20)
        plt.grid()

        plt.subplot(143)
        plt.plot(nrj, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('Decreasing energy', fontsize=20)
        plt.grid()
        
        plt.subplot(144)
        plt.plot(par_reg, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('$\lambda(m)$', fontsize=20)
        plt.title('Decreasing reg par', fontsize=20)
        plt.grid()
        
#############################################################################
#############################################################################
#############################################################################
#############################################################################

def save_cell_code_to_file(notebook_filename, cell_number, output_filename):
    # Load the notebook
    with open(notebook_filename, 'r') as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=nbformat.NO_CONVERT)

    # Get the code from the specified cell
    code = notebook_content['cells'][cell_number]['source']

    # Write the code to the output file
    with open(output_filename, 'w') as output_file:
        output_file.write(code)

#############################################################################
#############################################################################
#############################################################################
#############################################################################