from measure_2d import*
import scipy
import matplotlib.pyplot as plt


def Boosted_SFW_L2_2D(acquis, bg, sigma, X_domain, Y_domain, X_big, Y_big, ker='gaussian', m_0=0, par_reg=1e-5, nIter=5, BoostPar = float('inf'), verbose=False, plots=False, gt=0):
    
    ####################################################################
    ####################################################################
    
    def frank_wolfe_step(m,x_new,max_eta):
        a_k = m.a
        x_k = m.x
        N_k = m.N
        # I add the new position to the reconstruction
        if x_k.size == 0:
            x_k = np.vstack([x_new])
        else:
            x_k = np.vstack([x_k, x_new])
        if verbose:
            print('FRANK-WOLFE (NON SLIDING) STEP ')
            print(f'New position x^* = {x_new} max value of certificate {np.round(max_eta, 4)}')
            print('Measure_k : ')
            print('* x_k : ' + str(np.round(x_k, 4)))
        # define the lasso functional at each step: dimensions change! x_k is bigger at each step
        def lasso(a):
            aus = phi_vector(a, x_k, X_domain, Y_domain, sigma) 
            fidelity = 0.5*np.linalg.norm(acquis - aus)
            penalty = par_reg * np.linalg.norm(a, 1)
            return (fidelity + penalty)
        # solve the lasso minimisation problem
        init_guess = np.append(a_k, 0)
        res = scipy.optimize.minimize(lasso, init_guess)
        a_k = res.x.tolist()
        if verbose:
            print('* a_k : ' + str(np.round(a_k, 4)))
        m = Measure2D(a_k, x_k)
        # PRUNING
        m = m.prune()
        if verbose:
            print('Pruning')
            print('* x_k : ' + str(np.round(m.x, 4)))
            print('* a_k : ' + str(np.round(m.a, 4)))
        return m
    
    def sliding_step(m):
        if verbose:
            print('SLIDING STEP')
        a_k = m.a
        x_k = m.x
        N_k = m.N
        initial_guess = np.append(a_k, np.reshape(x_k, -1))
        # define the ''lasso double'' functional: it depends both on a and on x
        def lasso_double(params):
            a_p = params[:int(len(params)/3)] # Bout de code immonde, à corriger !
            x_p = params[int(len(params)/3):]
            x_p = x_p.reshape((len(a_p), 2))
            aus = phi_vector(a_p, x_p, X_domain, Y_domain, sigma)
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
                                                      Y_domain - x_p[i, 1], sigma))
                partial_a[i] = par_reg - integ/np.shape(X_domain)[0]

                grad_gauss_x = grad_x_gaussian_2D(X_domain - x_p[i, 0],
                                                    Y_domain - x_p[i, 1],
                                                    X_domain - x_p[i, 0], sigma)
                integ_x = np.sum(residual*grad_gauss_x) / (2*np.shape(X_domain)[0])
                partial_x[2*i] = a_p[i] * integ_x
                grad_gauss_y = grad_y_gaussian_2D(X_domain - x_p[i, 0],
                                                    Y_domain - x_p[i, 1],
                                                    Y_domain - x_p[i, 1], sigma)
                integ_y = np.sum(residual*grad_gauss_y)
                partial_x[2*i+1] = a_p[i] * integ_y / (2*np.shape(X_domain)[0])

            return(partial_a + partial_x)
        # solve the double optimisation problem 
        res = scipy.optimize.minimize(lasso_double, initial_guess,
                                      method='BFGS',
                                      jac=grad_lasso_double)
        a_k = (res.x[:int(len(res.x)/3)])
        x_k = (res.x[int(len(res.x)/3):]).reshape((len(a_k), 2))
        # compute the corresponding ''acquired measure''
        m = Measure2D(a_k, x_k)

        if verbose:
            print('Measure_k : ')
            print('* x_k : ' + str(np.round(x_k, 4)))
            print('* a_k : ' + str(np.round(a_k, 4)))

        m = m.prune()
        if verbose:
            print('Pruning')
            print('* x_k : ' + str(np.round(m.x, 4)))
            print('* a_k : ' + str(np.round(m.a, 4)))
        return m
    
    ####################################################################
    ####################################################################
    
    # initialisation
    N_ech_y = len(acquis)
    if m_0==0:   
        m_0 = Measure2D([], []) # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure2D(a_k, x_k)
    energy_vector = np.zeros(nIter)
    
    # COMPUTE THE CERTIFICATE
    eta_V_k = etak_L2(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker)
    x_star_index = np.unravel_index(np.argmax(eta_V_k, axis=None), eta_V_k.shape)
    x_star = np.array(x_star_index)[::-1]/N_ech_y 
    eta_max = np.max(eta_V_k)
    if verbose:
        print('Certificate argmax = ' + str(eta_max))
    x_star_old = x_star
    eta_max_old = eta_max
    
    for k in range(nIter):
        if verbose:
            print('\n' + '*** Iteration n. ' + str(k + 1) + ' ***')
        # Check on the max value of the certificate: if >1 add new spike
        ########## FRANK-WOLFE STEP #########
        if eta_max >= 1:
            # if it is the last iteration I perform a sliding step anyway
            # I perform a sliding step every BoostPar step
            if k == nIter -1:
                if measure_k.N>0:
                    if verbose:
                        print('Certificate argmax = ' + str(eta_max))
                        print('Sliding performed at the last iteration')
                    measure_k = frank_wolfe_step(measure_k,x_star,eta_max)
                    measure_k = sliding_step(measure_k)
                if verbose:
                    print("\n---- Stopping criterion: maximum number of iterations ----")
            elif (k+1) % BoostPar == 0:
                if verbose:
                    print('Certificate argmax = ' + str(eta_max))
                    print('Sliding step performed every ' + str(BoostPar) + ' iterations')
                measure_k = frank_wolfe_step(measure_k,x_star,eta_max)                
                measure_k = sliding_step(measure_k)
                # COMPUTE THE CERTIFICATE
                eta_V_k = etak_L2(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker)
                x_star_index = np.unravel_index(np.argmax(eta_V_k, axis=None), eta_V_k.shape)
                x_star = np.array(x_star_index)[::-1]/N_ech_y 
                eta_max = np.max(eta_V_k)
                if np.array_equal(x_star_old, x_star) and eta_max_old == eta_max: 
                    if verbose:
                        print('Certificate argmax = ' + str(eta_max))
                        print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                        print('----For loop stopped to avoid infinity loop----')
                    return(measure_k, energy_vector[:k])
                else: 
                    eta_max_old = eta_max
                    x_star_old = x_star
            else: 
                if verbose:
                    print('Certificate argmax = ' + str(eta_max))
                measure_k = frank_wolfe_step(measure_k,x_star,eta_max)
                # COMPUTE THE CERTIFICATE
                eta_V_k = etak_L2(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker)
                x_star_index = np.unravel_index(np.argmax(eta_V_k, axis=None), eta_V_k.shape)
                x_star = np.array(x_star_index)[::-1]/N_ech_y 
                eta_max = np.max(eta_V_k)
                if np.array_equal(x_star_old, x_star) and eta_max_old == eta_max: 
                    if verbose:
                        print('Certificate argmax = ' + str(eta_max))
                        print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                        print('----For loop stopped to avoid infinity loop----')
                    return(measure_k, energy_vector[:k])
                else: 
                    eta_max_old = eta_max
                    x_star_old = x_star
                

        # Check on the max value of the certificate: if <1 SLIDING or STOP
        elif eta_max < 1:

            # if <1 at first iteration -> STOP
            if k == 0:
                energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
                if verbose:
                    print('Certificate argmax = ' + str(eta_max))
                    print(f'* Energy : {energy_vector[k]:.3f}')
                    print("\n---- Stopping criterion: based on the certificate ----")
                return (measure_k, energy_vector[:k])

            # if <1 at later iterations -> SLIDING
            
            else:
                ####### SLIDING ########
                if verbose:
                    print('Certificate argmax = ' + str(eta_max))
                measure_k = sliding_step(measure_k)
                # COMPUTE THE CERTIFICATE
                eta_V_k = etak_L2(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, X_big, Y_big,ker=ker)
                x_star_index = np.unravel_index(np.argmax(eta_V_k, axis=None), eta_V_k.shape)
                x_star = np.array(x_star_index)[::-1]/N_ech_y 
                eta_max = np.max(eta_V_k)
                if eta_max < 1: 
                    energy_vector[k:] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
                    if verbose:
                        print('Certificate argmax = ' + str(eta_max))
                        print(f'* Energy : {energy_vector[k]:.3f}')
                        print("\n---- Stopping criterion: based on the certificate ----")
                    return (measure_k, energy_vector[:k])
                if np.array_equal(x_star_old, x_star) and eta_max_old == eta_max: 
                    if verbose:
                        print('Certificate argmax = ' + str(eta_max))
                        print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                        print('----For loop stopped to avoid infinity loop----')
                    return(measure_k, energy_vector[:k])
                else: 
                    eta_max_old = eta_max
                    x_star_old = x_star
                    
        if plots and measure_k.N>0:
            plt.figure(figsize=(4,4))
            cont = plt.contourf(X_domain, Y_domain, acquis, 100)
            for c in cont.collections:
                c.set_edgecolor("face")
            if gt!=0:
                plt.scatter(gt.x[:,0],gt.x[:,1], c='white')
            plt.scatter(measure_k.x[:,0],measure_k.x[:,1], c='red')
            plt.show()
        # reconstruction at iteration k
        a_k = measure_k.a
        x_k = measure_k.x
        Nk = measure_k.N
        energy_vector[k] = L2TV_cost_funct(measure_k, acquis, bg, par_reg, sigma, X_domain, Y_domain, ker=ker)
        if verbose:
            print(f'* Energy : {energy_vector[k]:.3f}')

        

    if verbose:
        print("\n---- Stopping criterion: End for ----")

    if verbose:
        print('\n' + 'L2-BLASSO with ' + str(k + 1) + ' iterations of Boosted SFW')
        print('* Regularisation parameter Lambda = : ' + str(par_reg))
        print('Reconstruction ')
        print('* positions: ' + str(np.round(measure_k.x, 4)))
        print('* amplitudes : ' + str(np.round(measure_k.a, 4)))
        if gt!=0:
            print('Ground truth ')
            print('* positions: ' + str(np.round(gt.x, 4)))
            print('* amplitudes : ' + str(np.round(gt.a, 4)))

    return (measure_k, energy_vector)
