from measure_2d import*
# from SFW_L2_2D import*
import scipy
import matplotlib.pyplot as plt


def SFW_L2_2D(acquis, bg, sigma, X_domain, Y_domain, X_big, Y_big, ker='gaussian', 
        m_0=0, par_reg=1e-5, nIter=5, sliding = 1, pruning = 1, 
        verbose = False, plots = False, gt = 0):
    
    def is_close_to(x_new, x_vec, delta):
        for entry in x_vec:
            if np.allclose(entry, x_new, atol=delta):
                return True
        return False
    
    # initialisation
    N_ech_y = len(acquis)
    if m_0==0:   
        m_0 = Measure2D([], []) # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure2D(a_k, x_k)
    energy_vector = np.zeros(nIter)
    fidelity_vector = np.zeros(nIter)
    penalty_vector = np.zeros(nIter)
    x_star_old = []
    eta_max_old = []
        
    for k in range(nIter):
        if verbose:
            print('\n' + 'Iteration n. ' + str(k+1))
        
        # step 1: compute the position for the new spike
        eta_V_k = etak_L2(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None), eta_V_k.shape)
        x_star = np.array(x_star_index)[::-1]/N_ech_y 
        eta_max = np.abs(eta_V_k[x_star_index])
                
        if np.array_equal(x_star_old, x_star) and eta_max_old == eta_max: 
            if verbose:
                print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                print('----For loop stopped to avoid infinity loop----')
            return(measure_k, energy_vector[:k+1],fidelity_vector[:k+1],penalty_vector[:k+1])
        else: 
            eta_max_old = eta_max
            x_star_old = x_star
            
        if verbose: 
            print(f'New position x^* index {x_star} max value of certificate = {np.round(eta_max, 2)}')
        
        # Check on the max value of the certificate: if <1 stop
        if eta_max < 1:
            energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
            fidelity_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, 0, sigma, X_domain, Y_domain, ker=ker)
            penalty_vector[k] = measure_k.tv()
            if verbose:
                print(f'* Energy: {energy_vector[k]:.3f}')
                print("\n---- Stopping criterion: based on the certificate ----")
            return(measure_k, energy_vector[:k+1],fidelity_vector[:k+1],penalty_vector[:k+1])
        
        elif is_close_to(x_star,x_k,1e-2):
            
            if verbose:
                print('New position is already present in the estimated spikes positions.')

            def lasso_double(params):
                a_p = params[:int(len(params)/3)] # Bout de code immonde, à corriger !
                x_p = params[int(len(params)/3):]
                x_p = x_p.reshape((len(a_p), 2))
                aus = phi_vector(a_p, x_p, X_domain, Y_domain, sigma)
                fidelity = 0.5*np.linalg.norm(acquis - aus)
                penalty = par_reg*np.linalg.norm(a_p, 1)
                return(fidelity + penalty)

            initial_guess = np.append(a_k, np.reshape(x_k, -1))
            bnds_a = [(0, np.inf) for _ in range(Nk)]
            bnds_x = [(0, 1) for _ in range(2*Nk)]
            bnds_new = bnds_a+bnds_x
            # solve the double optimisation problem
            res = scipy.optimize.minimize(lasso_double, initial_guess,
                                          method='L-BFGS-B', 
                                          bounds=bnds_new,
                                          options={'disp': True})
            a_k_plus = (res.x[:int(len(res.x)/3)])
            x_k_plus = (res.x[int(len(res.x)/3):]).reshape((len(a_k_plus), 2))
            # compute the corresponding ''acquired measure''
            measure_k = Measure2D(a_k_plus, x_k_plus)
            if verbose: 
                print('Sliding step performed at iteration ' + str(k+1))
                print('* x_k : ' + str(np.round(measure_k.x, 4)))
                print('* a_k : ' + str(np.round(measure_k.a, 4)))

            # get rid of null diracs with prune
            if pruning == 1:
                measure_k = measure_k.prune()
                if verbose:
                    print('Pruning performed at iteration ' + str(k+1))
                    print('* x_k : ' + str(np.round(measure_k.x, 4)))
                    print('* a_k : ' + str(np.round(measure_k.a, 4)))
                    
                    
        else:
            measure_k_demi = Measure2D()
            if x_k.size == 0:
                x_k_demi = np.vstack([x_star])
            else:
                x_k_demi = np.vstack([x_k, x_star])

            # define the lasso functional at each step: dimensions change! x_k_demi is bigger at each step
            def lasso(a):
                aus = phi_vector(a, x_k_demi, X_domain, Y_domain, sigma)+bg
                fidelity = 0.5*np.linalg.norm(acquis - aus)
                penalty = par_reg*np.linalg.norm(a, 1)
                return(fidelity + penalty)
            init_guess = np.append(a_k, 0)
            res = scipy.optimize.minimize(lasso, init_guess)
            a_k_demi = res.x
            measure_k_demi += Measure2D(a_k_demi,x_k_demi)
            if verbose:
                print('Insertion step')
                print('* a_k_demi : ' + str(np.round(a_k_demi, 2))) 
                print('* x_k_demi : ' + str(np.round(x_k_demi, 2)))
            if plots and measure_k_demi.N>0:
                plt.figure(figsize=(4,4))
                cont = plt.contourf(X_domain, Y_domain, acquis, 100)
                for c in cont.collections:
                    c.set_edgecolor("face")
                if gt!=0:
                    plt.scatter(gt.x[:,0],gt.x[:,1], c='white')
                plt.scatter(measure_k_demi.x[:,0],measure_k_demi.x[:,1], c='red')
                plt.show()

            if sliding == 1:
                # define the ''lasso double'' functional: it depends both on a and on x
                def lasso_double(params):
                    a_p = params[:int(len(params)/3)] # Bout de code immonde, à corriger !
                    x_p = params[int(len(params)/3):]
                    x_p = x_p.reshape((len(a_p), 2))
                    aus = phi_vector(a_p, x_p, X_domain, Y_domain, sigma)+bg
                    fidelity = 0.5*np.linalg.norm(acquis - aus)
                    penalty = par_reg*np.linalg.norm(a_p, 1)
                    return(fidelity + penalty)

                def grad_lasso_double(params):
                    a_p = params[:int(len(params)/3)]
                    x_p = params[int(len(params)/3):]
                    x_p = x_p.reshape((len(a_p), 2))
                    N = len(a_p)
                    partial_a = N*[0]
                    partial_x = 2*N*[0]
                    residual = acquis - phi_vector(a_p, x_p, X_domain, Y_domain, sigma)-bg
                    for i in range(N):
                        integ = np.sum(residual*gaussian_2D(X_domain - x_p[i, 0],
                                                              Y_domain - x_p[i, 1]))
                        partial_a[i] = par_reg - integ/np.shape(X_domain)[0]

                        grad_gauss_x = grad_x_gaussian_2D(X_domain - x_p[i, 0],
                                                            Y_domain - x_p[i, 1],
                                                            X_domain - x_p[i, 0])
                        integ_x = np.sum(residual*grad_gauss_x) / (2*np.shape(X_domain)[0])
                        partial_x[2*i] = a_p[i] * integ_x
                        grad_gauss_y = grad_y_gaussian_2D(X_domain - x_p[i, 0],
                                                            Y_domain - x_p[i, 1],
                                                            Y_domain - x_p[i, 1])
                        integ_y = np.sum(residual*grad_gauss_y)
                        partial_x[2*i+1] = a_p[i] * integ_y / (2*np.shape(X_domain)[0])

                    return(partial_a + partial_x)

                # solve the double optimisation problem
                initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))
                res = scipy.optimize.minimize(lasso_double, initial_guess,
                                              method='BFGS')
                                          # jac=grad_lasso_double,
#                                           options={'disp': __deboggage__})
                a_k_plus = (res.x[:int(len(res.x)/3)])
                x_k_plus = (res.x[int(len(res.x)/3):]).reshape((len(a_k_plus), 2))
                # compute the corresponding ''acquired measure''
                measure_k = Measure2D(a_k_plus, x_k_plus)
                if verbose: 
                    print('Sliding step performed at iteration ' + str(k+1))
                    print('* x_k : ' + str(np.round(measure_k.x, 4)))
                    print('* a_k : ' + str(np.round(measure_k.a, 4)))
            ###### NO SLIDING #######
            elif sliding == 0:
                measure_k = measure_k_demi
                if verbose: 
                    print('No sliding step')
            
            
            # get rid of null diracs with prune
            if pruning == 1:
                measure_k = measure_k.prune() 
                if verbose: 
                    print('Pruning performed at iteration ' + str(k+1))
                    print('* x_k : ' + str(np.round(measure_k.x, 4)))
                    print('* a_k : ' + str(np.round(measure_k.a, 4)))
                    
            a_k = measure_k.a
            x_k = measure_k.x
            Nk = measure_k.N
            energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
            fidelity_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, 0, sigma, X_domain, Y_domain, ker=ker)
            penalty_vector[k] = measure_k.tv()
            if verbose: 
                print(f'* Energy : {energy_vector[k]:.3f}')
            
            if plots and measure_k.N>0:
                plt.figure(figsize=(4,4))
                cont = plt.contourf(X_domain, Y_domain, acquis, 100)
                for c in cont.collections:
                    c.set_edgecolor("face")
                if gt!=0:
                    plt.scatter(gt.x[:,0],gt.x[:,1], c='white')
                plt.scatter(measure_k.x[:,0],measure_k.x[:,1], c='red')
                plt.show()
            
            
    if verbose:       
        print("\n---- Stopping criterion: End for ----")
    return(measure_k, energy_vector, fidelity_vector, penalty_vector)



def Homotopy_SFW_L2_2D(acquis, bg, sigma, sigma_target, X_domain, Y_domain, X_big, Y_big, ker='gaussian', 
        m_0=0, nnIter=5, nIter=1, c=0.5, sliding = 1, pruning = 1, 
        verbose = False, plots = False, gt = 0):
    
    def fidelity(acquis,bg,m):
        aus = m.kernel(X_domain, Y_domain, sigma, ker=ker)+bg
        return 0.5*np.linalg.norm(acquis - aus)
    
    # initialisation
    N_ech_y = len(acquis)
    if m_0==0:   
        m_0 = Measure2D([], []) # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure2D(a_k, x_k)
    energy_vector = np.zeros(nnIter)
    lambda_sequence = np.zeros(nnIter)
    sigma_sequence = np.zeros(nnIter)
    fidelity_vector = []
    penalty_vector = []
    
    if verbose: 
        print('sigma_target = '+str(sigma_target))
    
    # compute lambda_0 
    eta_max = np.max( np.abs(etak_L2(measure_k, acquis, bg, 5, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker)))
    par_reg = eta_max
    
    for k in range(nnIter): 
        if verbose: 
            print('\n\n*** Homotopy iteration n.'+str(k)+' ***')
            print('max of certificate='+str(eta_max))
            print('lambda='+str(par_reg))
        
        (measure_k, nrj, fid, pen) = SFW_L2_2D(acquis, bg, sigma, X_domain, Y_domain, X_big, 
                                           Y_big, ker=ker, m_0=measure_k, par_reg=par_reg, 
                                           nIter=nIter, sliding = 1, pruning = 1, 
                                           verbose=verbose, plots = plots, gt = gt)
        
#         fidelity_vector = np.append(fidelity_vector,fid)
#         penalty_vector = np.append(penalty_vector,pen)
        fidelity_vector.append(fid)
        penalty_vector.append(pen)
        
        sigma_t = fidelity(acquis,bg,measure_k)
        sigma_sequence[k] = sigma_t
        if verbose: 
            print('sigma_t='+str(sigma_t))
            
        energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
        
        if sigma_t > sigma_target:
            eta_max = np.max(etak_L2(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker))
            par_reg = par_reg*eta_max/(c+1)
            lambda_sequence[k] = par_reg            
        else: 
            print("\n---- Stopping criterion based on the homotopy ----")
#             return(measure_k, energy_vector[:k+1], lambda_sequence[:k], sigma_sequence[:k+1], fidelity_vector, penalty_vector)
            return(measure_k, energy_vector[:k+1], lambda_sequence[:k], sigma_sequence[:k+1], fidelity_vector[:k+1], penalty_vector[:k+1])

        
        if verbose: 
            print(f'* Energy : {energy_vector[k]:.3f}')

    print("\n---- End homotopy for: stopping criterion based on the max number of iterations ----")
    return(measure_k, energy_vector, lambda_sequence, sigma_sequence, fidelity_vector, penalty_vector)
    

