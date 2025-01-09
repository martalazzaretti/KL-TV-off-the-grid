from measure_3d import*
from Boosted_SFW_L2_3D import*
import scipy
import matplotlib.pyplot as plt

def Homotopy_Boosted_SFW_L2_3D(acquis, bg, sigma_target, sig_x, sig_y, sig_z, X_domain, Y_domain, Z_domain, N_xy, N_z, 
                                      pixel_size_xy, pixel_size_z, 
        m_0=0, nnIter=5, nIter=1, c=5, BoostPar = float('inf'), verbose = False, plots = False, gt = 0):
    
    def fidelity(acquis,bg,m):
        aus = m.kernel(X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z)
        return 0.5*my3dnorm(acquis - aus)
    
    # initialisation
    N_ech_y = len(acquis)
    if m_0==0:   
        m_0 = Measure3D([], []) # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure3D(a_k, x_k)
    energy_vector = np.zeros(nnIter)
    lambda_sequence = np.zeros(nnIter)
    
    if verbose: 
        print('sigma_target = '+str(sigma_target))
    
    # compute lambda_0 
    eta_max = np.max( np.abs(etak_KL(measure_k, acquis, bg, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, 5)))
    par_reg = eta_max
    
    for k in range(nnIter): 
        if verbose: 
            print('\n\n*** Homotopy iteration n.'+str(k)+' ***')
            print('max of certificate='+str(eta_max))
            print('lambda='+str(par_reg))
        
        (measure_k, nrj) = Boosted_SFW_L2_3D(acquis, bg, sig_x, sig_y, sig_z, X_domain ,Y_domain , Z_domain,  N_xy, N_z, 
                        pixel_size_xy, pixel_size_z, par_reg=par_reg, nIter=nIter, m_0=measure_k, BoostPar = BoostPar, 
                                           verbose=verbose, plots=plots)
        
        sigma_t = fidelity(acquis,bg,measure_k)
        if verbose: 
            print('sigma_t='+str(sigma_t))
            
        energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg,  X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, par_reg)
        
        if sigma_t > sigma_target:
            eta_max = np.max(etak_L2(measure_k, acquis, bg, X_domain, Y_domain, Z_domain, sig_x, sig_y, sig_z, par_reg))
            par_reg = par_reg*eta_max/(c+1)
            lambda_sequence[k] = par_reg            
        else: 
            print("\n---- Stopping criterion based on the homotopy ----")
            return(measure_k, energy_vector[:k], lambda_sequence[:k])

        
        if verbose: 
            print(f'* Energy : {energy_vector[k]:.3f}')

    print("\n---- End homotopy for: stopping criterion based on the max number of iterations ----")
    return(measure_k, energy_vector, lambda_sequence)
    

