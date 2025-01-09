import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import random

#############################################################################
#############################################################################
#############################################################################
#############################################################################

#######################################################
#######################################################

def my3dnorm(x,p=2):
    if p==2:
        return np.sqrt(np.sum(np.square(x[:,:,:])))
    else: 
        return np.sqrt(np.sum(np.power(x[:,:,:],p)))
    
#######################################################
#######################################################

def gaussian_3D(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z):
    '''Guassian 3D centered in 0'''
    expo = np.exp(-np.power(X_domain, 2)/(2*sig_x**2) - np.power(Y_domain, 2)/(2*sig_y**2) - np.power(Z_domain, 2)/(2*sig_z**2))
#     normalis = sig_x*sig_y*sig_z * np.sqrt((2*np.pi)**3)
    return expo #/normalis

def grad_x_gaussian_3D(X_domain, Y_domain, Z_domain, X_deriv, sig_x, sig_y, sig_z):
    '''Guassian 3D centered in 0. Attention, no chain 
    rule derivation'''
    expo = np.exp(-np.power(X_domain, 2)/(2*sig_x**2) - np.power(Y_domain, 2)/(2*sig_y**2) - np.power(Z_domain, 2)/(2*sig_z**2))
#     normalis = sig_x*sig_y*sig_z * np.sqrt((2*np.pi)**3)
    c = - X_deriv * (2*sig_x**2)
    return c * expo #/ normalis

def grad_y_gaussian_3D(X_domain, Y_domain, Z_domain, Y_deriv, sig_x, sig_y, sig_z):
    '''Guassian 3D centered in 0. Attention, no chain 
    rule derivation'''
    expo = np.exp(-np.power(X_domain, 2)/(2*sig_x**2) - np.power(Y_domain, 2)/(2*sig_y**2) - np.power(Z_domain, 2)/(2*sig_z**2))
#     normalis = sig_x*sig_y*sig_z * np.sqrt((2*np.pi)**3) 
    c = - Y_deriv * (2*sig_y**2)
    return c * expo #/ normalis

def grad_z_gaussian_3D(X_domain, Y_domain, Z_domain, Z_deriv, sig_x, sig_y, sig_z):
    '''Guassian 3D centered in 0. Attention, no chain 
    rule derivation'''
    expo = np.exp(-np.power(X_domain, 2)/(2*sig_x**2) - np.power(Y_domain, 2)/(2*sig_y**2) - np.power(Z_domain, 2)/(2*sig_z**2))
#     normalis = sig_x*sig_y*sig_z * np.sqrt((2*np.pi)**3) 
    c = - Z_deriv * (2*sig_z**2)
    return c * expo # / normalis

#######################################################
#######################################################

#############################################################################
#############################################################################
#############################################################################
#############################################################################


class Measure3D:
    def __init__(self, amplitude=[], position=[]):
        if len(amplitude) != len(position):
            raise ValueError('Not the same number of amplitudes and positions')
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
        return Measure3D(a_new, x_new)


    def __eq__(self, m):
        if m == 0 and (self.a == [] and self.x == []):
            return True
        elif isinstance(m, self.__class__):
            return self.__dict__ == m.__dict__
        else:
            return False


    def kernel(self, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z):
        '''Convolution with blurring kernel'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X_domain*0
        for i in range(0,N):
            D = gaussian_3D(X_domain - x[i,0], Y_domain - x[i,1], Z_domain - x[i,2], sig_x, sig_y, sig_z)
            acquis += a[i]*D
        return acquis


    def acquisition(self, X_domain, Y_domain, Z_domain, N_xy, N_z, sig_x, sig_y, sig_z, nv):
        w = nv*np.random.standard_normal((N_xy, N_xy, N_z))
        acquis = self.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z) + w
        return acquis

    def tv(self):
        '''Total variation of the measure: \ell_1 norm of the amplitudes vector'''
        try:
            return np.linalg.norm(self.a, 1)
        except ValueError:
            return 0

        
#     def energy(self, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, y, reg_par):
#         fidelity = 0.5*my3dnorm(y - self.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z),2)**2
#         penalty = reg_par*self.tv()
#         return(fidelity + penalty)


    def prune(self, tol=1e-3):
        '''Remove the spikes with 0 amplitudes'''
        # nnz = np.count_nonzero(self.a)
        nnz_a = np.array(self.a)
        nnz = np.abs(nnz_a) > tol
        nnz_a = nnz_a[nnz]
        nnz_x = np.array(self.x)
        nnz_x = nnz_x[nnz]
        m = Measure3D(nnz_a, nnz_x)
        return m
    
#############################################################################
#############################################################################
#############################################################################
#############################################################################

def phi(m, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z):
    return m.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)

def phi_vector(a, x, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z):
    m_tmp = Measure3D(a, x)
    return(m_tmp.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z))

def phiAdjoint(acquis, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z):
    '''Adjoint operator'''
    out = gaussian_3D(X_domain, Y_domain,Z_domain, sig_x, sig_y, sig_z)
    eta = scipy.signal.convolve(out, acquis, mode='same') #/(N_xy**2*N_z)
    return eta


#############################################################################
#############################################################################
#############################################################################
#############################################################################

def rand_gt(sizeGrid_xy, sizeGrid_z, pixel_size_xy, pixel_size_z, N_mol, margin=0):
    '''margin is given in the same measure unit as sizegrid'''
    m = Measure3D(np.empty((0,0)), np.empty((0,0)))  
    for l in np.arange(N_mol):
        if pixel_size_xy!=1:
            i = random.randrange(-sizeGrid_xy/2+margin, sizeGrid_xy/2-margin)
            j = random.randrange(-sizeGrid_xy/2+margin, sizeGrid_xy/2-margin)
        elif pixel_size_xy==1:
            i = random.randint(-sizeGrid_xy/2+margin, sizeGrid_xy/2-margin)
            j = random.randint(-sizeGrid_xy/2+margin, sizeGrid_xy/2-margin)
        if pixel_size_z!=1:
            k = random.randrange(-sizeGrid_z/2+margin, sizeGrid_z/2-margin)
        elif pixel_size_z==1:
             k = random.randint(-sizeGrid_z/2+margin, sizeGrid_z/2-margin)
        x = [i,j,k]
        a = random.uniform(0.6,1.4)        
        m1 = Measure3D([a],[x])
        m = m+m1
    return m 

#############################################################################
#############################################################################
#############################################################################
#############################################################################

def etak_L2(measure, y, bg,  X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, par_reg):
    eta = 1/par_reg*phiAdjoint(y - phi(measure, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)-bg, 
                             X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)
    return eta

def L2TV_cost_funct(m, y, bg,  X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, par_reg):
    aus = m.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z) + bg
    fidelity = 0.5*my3dnorm(y - aus)
    penalty = par_reg*m.tv()
    return(fidelity + penalty)


def KLTV_cost_funct(m, y, bg,  X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, par_reg):
    aus = m.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)
    fidelity = np.sum(aus+bg-y+y*np.log(y) - y*np.log(aus+bg))
    penalty = par_reg*m.tv()
    return(fidelity + penalty)


def etak_KL(measure, y, bg,  X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, par_reg):
    eta = np.ones(np.shape(y))-np.divide(y,phi(measure, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)+bg)
    eta = -1/par_reg*phiAdjoint(eta, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)
    return eta


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





