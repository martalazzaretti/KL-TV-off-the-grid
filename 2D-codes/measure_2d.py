import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

#############################################################################
#############################################################################
#############################################################################
#############################################################################


def gaussian(domain, sigma):
    '''Gaussian centered in zero'''
    expo = np.exp(-np.power(domain, 2)/(2*sigma**2))
    normalis = sigma * (2*np.pi)
    return expo/normalis

def gaussian_2D(X_domain, Y_domain, sigma):
    '''2D Gaussian centered in zero'''
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma**2))
    normalis = sigma * (2*np.pi)
    return expo/normalis

def grad_x_gaussian_2D(X_domain, Y_domain, X_deriv, sigma):
    
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma**2))
    normalis = sigma_g**3 * (2*np.pi)
    expo = - X_deriv * expo
    return expo / normalis

def grad_y_gaussian_2D(X_domain, Y_domain, Y_deriv, sigma):
    
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma**2))
    normalis = sigma_g**3 * (2*np.pi)
    expo = - Y_deriv * expo
    return expo / normalis

#############################################################################
#############################################################################
#############################################################################
#############################################################################

class Measure2D:
    def __init__(self, amplitude=[], position=[]):
        if len(amplitude) != len(position):
            raise ValueError('Amplitudes and positions need to have the same lenght')
        if isinstance(amplitude, np.ndarray) and isinstance(position, np.ndarray):
            self.a = amplitude
            self.x = position
        else:
            self.x = np.array(position)
            self.a = np.array(amplitude)
        self.N = len(amplitude)


    def __add__(self, m):
        a_new = np.append(self.a, m.a)
        x_new = np.array(list(self.x) + list(m.x))
        return Measure2D(a_new, x_new)


    def __eq__(self, m):
        if m == 0 and (self.a == [] and self.x == []):
            return True
        elif isinstance(m, self.__class__):
            return self.__dict__ == m.__dict__
        else:
            return False

    def kernel(self, X_domain, Y_domain, sigma, ker='gaussian'):
        '''Kernel applied to the discrete measure, i.e. convolution with PSF'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X_domain*0
        if ker == 'gaussian':
            for i in range(0,N):
                D = np.sqrt(np.power(X_domain - x[i,0],2) + np.power(Y_domain - x[i,1],2))
                acquis += a[i]*gaussian(D, sigma)
            return acquis
        else:
            raise TypeError('Unknown kernel! You might want to implement it')


    def acquisition(self, X_domain, Y_domain, sigma, bg, nv, ker='gaussian', noiseType='Gaussian'):
        if noiseType == 'Gaussian':
            N = np.shape(X_domain)[0]
            w = nv*np.random.standard_normal((N, N))
            # w = nv*np.random.random_sample((N, N))
            acquis = self.kernel(X_domain, Y_domain, sigma, ker='gaussian') + bg + w
        elif noiseType == 'Poisson':
            acquis = self.kernel(X_domain, Y_domain, sigma, ker='gaussian') + bg
            acquis = np.random.poisson(acquis * nv) / nv
        else:
            raise TypeError('Unknown noiseType! You might want to implement it')
        return acquis

    def tv(self):
        '''TV norm of the measure'''
        try:
            return np.linalg.norm(self.a, 1)
        except ValueError:
            return 0

    def prune(self, tol=1e-3):
        '''Remove diracs with zero amplitude'''
        # nnz = np.count_nonzero(self.a)
        nnz_a = np.array(self.a)
        nnz = nnz_a > tol
        nnz_a = nnz_a[nnz]
        nnz_x = np.array(self.x)
        nnz_x = nnz_x[nnz]
        m = Measure2D(nnz_a, nnz_x)
        return m
    
    
#############################################################################
#############################################################################
#############################################################################
#############################################################################   
    
    
def phi(m, X_domain, Y_domain, sigma):
    return m.kernel(X_domain, Y_domain, sigma)

def phi_vector(a, x, X_domain, Y_domain, sigma):
    m_tmp = Measure2D(a, x)
    return(m_tmp.kernel(X_domain, Y_domain, sigma))

def phiAdjoint(acquis, X_domain, Y_domain, sigma, ker='gaussian'):
    '''Computation of the adjoint with convolution '''
    N = np.shape(X_domain)[0]
    out = gaussian_2D(X_domain, Y_domain, sigma)
    eta = scipy.signal.convolve2d(out, acquis, mode='valid')/(N**2)
    return eta

#############################################################################
#############################################################################
#############################################################################
#############################################################################


def random_measure_2d(N, margin):
    x = np.round(np.random.rand(N,2), 2)
    x = margin + x*(1-2*margin)
    a = np.round(0.5 + np.random.rand(1,N), 2)[0]
    return Measure2D(a, x)

#############################################################################
#############################################################################
#############################################################################
#############################################################################


def etak_L2(m, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big, ker='gaussian'):
    eta = 1/par_reg*phiAdjoint(acquis - phi(m, X_domain, Y_domain, sigma)-bg, 
                             X_big, Y_big, sigma, ker='gaussian')
    return eta

def L2TV_cost_funct(m, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker='gaussian'):
    aus = m.kernel(X_domain, Y_domain, sigma, ker=ker) + bg
    fidelity = 0.5*np.linalg.norm(acquis - aus)**2
    penalty = par_reg*m.tv()
    return(fidelity + penalty)


def KLTV_cost_funct(m, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker='gaussian'):
    aus = phi(m, X_domain, Y_domain, sigma)
    fidelity = np.sum(aus+bg-acquis+acquis*np.log(acquis) - acquis*np.log(aus+bg))
    penalty = par_reg*m.tv()
    return(fidelity + penalty)


def etak_KL(m, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big, ker='gaussian'):
    eta = np.ones(len(X_domain))-np.divide(acquis,phi(m, X_domain, Y_domain, sigma)+bg)
    eta = -1/par_reg*phiAdjoint(eta, X_big, Y_big, sigma, ker='gaussian')
    return eta

#############################################################################
#############################################################################
#############################################################################
#############################################################################


def plot_results(m, energy, acquis, X_domain, Y_domain, sigma, certificate, gt=0):
    if m.a.size > 0:
        fig = plt.figure(figsize=(15,12))
#         fig.suptitle(f'Reconstruction for $\lambda = {lambda_regul:.0e}$ ' + 
#                      f'and $\sigma_B = {niveau_bruits:.0e}$', fontsize=20)

        plt.subplot(221)
        cont1 = plt.contourf(X_domain, Y_domain, acquis, 100) #, cmap='seismic'
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar();
        if gt!=0:
            plt.scatter(gt.x[:,0], gt.x[:,1], marker='x', c='white', s=100,
                        label='GT spikes')
        plt.scatter(m.x[:,0], m.x[:,1], marker='+', c='red', s=100,
                    label='Recovered spikes')
        plt.legend(loc=2)

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Acquisition $y = \Phi m_{a_0,x_0} + w$', fontsize=20)

        plt.subplot(222)
        cont2 = plt.contourf(X_domain, Y_domain, m.kernel(X_domain, Y_domain, sigma), 100)#, cmap='seismic'
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Reconstruction $m_{a,x}$', fontsize=20)
        plt.colorbar();
        
        plt.subplot(223)
        cont3 = plt.contourf(X_domain, Y_domain, certificate, 100) #,cmap='seismic'
        for c in cont3.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Certificate $\eta_V$', fontsize=20)
        plt.colorbar();

        plt.subplot(224)
        plt.plot(energy, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('BLASSO energy $T_\lambda(m)$', fontsize=20)
        plt.grid()
#         if __saveFig__:
#             plt.savefig('fig/dirac-certificat-2d.pdf', format='pdf', dpi=1000,
#                         bbox_inches='tight', pad_inches=0.03)


#############################################################################
#############################################################################
#############################################################################
#############################################################################

def jaccard_rmse(gt, rec, tol):
    """
    Compute Jaccard index and related metrics for 2D arrays.
    Save in CD_rec_ampl and CD_rec_pos the amplitudes and positions 
    of the reconstructed molecules classified as correctly detected;
    and save in CD_gt_ampl and CD_gt_pos the amplitudes and positions 
    of the ground truth molecules classified as correctly detected.
    Compute RMSE for amplitudes and positions, only using the correctly 
    detected molecules saved as above. 

    Parameters:
    gt (dict): Ground truth dictionary with 'x' and 'a' keys, where 'x' is a 2D array of positions and 'a' is a 1D array of amplitudes.
    rec (dict): Reconstructed dictionary with 'x' and 'a' keys, where 'x' is a 2D array of positions and 'a' is a 1D array of amplitudes.
    tol (float): Tolerance for set element comparison.

    Returns:
    tuple: Jaccard index, true positives, false negatives, false positives,
           RMSE for amplitudes, RMSE for positions.
    """
    
    # Helper function to determine if elements are within tolerance
    within_tol = lambda a, b: np.all(np.abs(a - b) <= tol)
    
    if rec.N!=0:
        # Convert arrays to sets
        set_gt = set(map(tuple, gt.x))
        set_rec = set(map(tuple, rec.x))

        CD_gt_pos = []
        CD_rec_pos = []

        while len(set_gt) > 0 and len(set_rec) > 0:
            # Compute the absolute differences
            dist_array = np.linalg.norm(np.array(list(set_gt))[:, np.newaxis] - np.array(list(set_rec)), axis=2)
            min_indices = np.unravel_index(np.argmin(dist_array), dist_array.shape)
            min_index_gt, min_index_rec = min_indices

            a = np.array(list(set_gt))[min_index_gt]
            b = np.array(list(set_rec))[min_index_rec]

            if within_tol(a, b):
                CD_gt_pos.append(a)
                CD_rec_pos.append(b)
                set_gt.remove(tuple(a))
                set_rec.remove(tuple(b))
            else:
                break  # None of the pairings has a distance smaller than the tolerance

        CD_gt_pos = np.array(CD_gt_pos)
        CD_rec_pos = np.array(CD_rec_pos)
        
        # Compute correctly detected molecules
        true_positives = len(CD_gt_pos)
        false_negatives = len(gt.x) - true_positives
        false_positives = len(rec.x) - true_positives

        jaccard_index = true_positives / (true_positives + false_negatives + false_positives)
        
        if len(CD_gt_pos)>0 and len(CD_rec_pos)>0:
            # Save correctly detected amplitudes 
            CD_rec_ampl = np.array([rec.a[np.where(np.all(rec.x == pos, axis=1))[0][0]] for pos in CD_rec_pos])
            CD_gt_ampl = np.array([gt.a[np.where(np.all(gt.x == pos, axis=1))[0][0]] for pos in CD_gt_pos])

            # Compute RMSE for amplitudes and positions
            RMSE_ampl = np.sqrt(np.nanmean((CD_rec_ampl - CD_gt_ampl) ** 2))
            RMSE_pos = np.sqrt(np.nanmean(np.linalg.norm(CD_rec_pos - CD_gt_pos, axis=1) ** 2))
        else: 
            RMSE_ampl = 0 
            RMSE_pos = 0
    else: 
        jaccard_index = 0
        true_positives = 0
        false_negatives = len(gt.x)
        false_positives = 0 
        RMSE_ampl = 0 
        RMSE_pos = 0

    return jaccard_index, true_positives, false_negatives, false_positives, RMSE_ampl, RMSE_pos


