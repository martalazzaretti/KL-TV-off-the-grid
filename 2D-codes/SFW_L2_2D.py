from measure_2d import*
import scipy
import matplotlib.pyplot as plt



def SFW_L2_2D(acquis, bg, sigma, X_domain, Y_domain, X_big, Y_big, ker='gaussian', 
        m_0=0, par_reg=1e-5, nIter=5, sliding = 1, pruning = 1, 
        verbose = False, plots = False, gt = 0):
    
    # initialisation
    N_ech_y = len(acquis)
    if m_0==0:   
        m_0 = Measure2D([], []) # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure2D(a_k, x_k)
    energy_vector = np.zeros(nIter)
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
            return(measure_k, energy_vector[:k])
        else: 
            eta_max_old = eta_max
            x_star_old = x_star
            
        if verbose: 
            print(f'New position x^* index {x_star} max value of certificate = {np.round(eta_max, 2)}')
        
        # Check on the max value of the certificate: if <1 stop
        if eta_max < 1:
            energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
            if verbose:
                print(f'* Energy: {energy_vector[k]:.3f}')
                print("\n---- Stopping criterion: based on the certificate ----")
            return(measure_k, energy_vector)
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
                plt.axis('off')
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
                plt.axis('off')
                plt.show()
            
            
    if verbose:       
        print("\n---- Stopping criterion: End for ----")
    return(measure_k, energy_vector)
