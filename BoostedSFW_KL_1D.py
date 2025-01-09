from measure_1d import*
import scipy
import matplotlib.pyplot as plt


def Boosted_SFW_KULLBACK_1D(X, acquis, bg, sigma, ker='gaussian', m_0=0, par_reg=1e-5, nIter=5, BoostPar = float('inf'), verbose=False, plots=False, gt=0):
    
    ####################################################################
    ####################################################################
    
    def frank_wolfe_step(m,x_new,max_eta):
        a_k = m.a
        x_k = m.x
        N_k = m.N
        # I add the new position to the reconstruction
        x_k = x_k + [x_new]
        if verbose:
            print('FRANK-WOLFE (NON SLIDING) STEP ')
            print(f'New position x^* = {x_new} max value of certificate {np.round(max_eta, 4)}')
            print('Measure_k : ')
            print('* x_k : ' + str(np.round(x_k, 4)))
        # define the lasso functional at each step: dimensions change! x_k is bigger at each step
        def lasso(a):
            aus = phi_vector(a, x_k, X, sigma, ker=ker) 
            fidelity = np.sum(aus+bg-acquis+acquis*np.log(acquis) - acquis*np.log(aus+bg))
            penalty = par_reg * np.linalg.norm(a, 1)
            return (fidelity + penalty)
        def grad_lasso(a):
            grad = np.ones(len(X))-np.divide(acquis,phi_vector(a,x_k_demi,X,sigma,ker=ker)+bg)
            grad = phi_amplitude_adjoint(grad, x_k_demi, X, sigma, ker='gaussian')+par_reg*np.sign(a)
            return grad
        # solve the lasso minimisation problem 
        init_guess = a_k+[0]
        bnds = [(0, np.inf) for _ in range(N_k+1)]
        res = scipy.optimize.minimize(lasso, init_guess, method='L-BFGS-B', bounds=bnds)
        # res = scipy.optimize.minimize(lasso, np.ones(N_k+1), jac=grad_lasso, bounds=bnds)
        a_k = res.x.tolist()
        if verbose:
            print('* a_k : ' + str(np.round(a_k, 4)))
        m = Measure(a_k, x_k)
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
        initial_guess = a_k + x_k  # amplitude a che usiamo come inizializzazione sono nella prima metà della
                                   # lista, posizioni x nella seconda metà
        # define the ''lasso double'' functional: it depends both on a and on x
        def lasso_double(params):
            a = params[:int(len(params) / 2)]
            x = params[int(len(params) / 2):]
            aus = phi_vector(a,x,X,sigma,ker=ker)
            fidelity = np.sum(aus+bg-acquis+acquis*np.log(acquis) - acquis*np.log(aus+bg))
            penalty = par_reg*np.linalg.norm(a, 1)
            return(fidelity + penalty)
        # bounds
        bnds_a = [(0, np.inf) for _ in range(N_k+1)]
        bnds_x = [(0, 1) for _ in range(N_k+1)]
        bnds_new = bnds_a + bnds_x
        # solve the double optimisation problem 
        res = scipy.optimize.minimize(lasso_double, initial_guess,
                                      method='L-BFGS-B', 
                                      bounds=bnds_new,
                                      options={'disp': True})
        a_k = (res.x[:int(len(res.x) / 2)]).tolist()
        x_k = (res.x[int(len(res.x) / 2):]).tolist()
        # compute the corresponding ''acquired measure''
        m = Measure(a_k, x_k)
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
    if m_0 == 0:
        m_0 = Measure([], [])  # empty discrete measure
    Nk = m_0.N  # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure(a_k, x_k)
    energy_vector = np.zeros(nIter)
    
    # COMPUTE THE CERTIFICATE
    eta_V_k = etak_KL(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
    x_star = np.argmax(eta_V_k)/len(eta_V_k)
    eta_max = np.max(eta_V_k)
    print('Certificate argmax = ' + str(eta_max))
    x_star_old = x_star
    eta_max_old = eta_max
    
    for k in range(nIter):     
        print('\n' + '*** Iteration n. ' + str(k + 1) + ' ***')
        # Check on the max value of the certificate: if >1 add new spike
        ########## FRANK-WOLFE STEP #########
        if eta_max >= 1:
            # if it is the last iteration I perform a sliding step anyway
            # I perform a sliding step every BoostPar step
            if k == nIter -1:
                if measure_k.N>0:
                    measure_k = frank_wolfe_step(measure_k,x_star,eta_max)
                    print('Sliding performed at the last iteration')
                    measure_k = sliding_step(measure_k)
                print("\n---- Stopping criterion: maximum number of iterations ----")
            elif (k+1) % BoostPar == 0: 
                measure_k = frank_wolfe_step(measure_k,x_star,eta_max)
                print('Sliding step performed every ' + str(BoostPar) + ' iterations')
                measure_k = sliding_step(measure_k)
                # COMPUTE THE CERTIFICATE
                eta_V_k = etak_KL(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
                x_star = np.argmax(eta_V_k)/len(eta_V_k)
                eta_max = np.max(eta_V_k)
                print('Certificate argmax = ' + str(eta_max))
                if x_star_old == x_star and eta_max_old == eta_max: 
                    if verbose:
                        print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                        print('----For loop stopped to avoid infinity loop----')
                    return(measure_k, energy_vector[:k])
                else: 
                    eta_max_old = eta_max
                    x_star_old = x_star
            else: 
                measure_k = frank_wolfe_step(measure_k,x_star,eta_max)
                # COMPUTE THE CERTIFICATE
                eta_V_k = etak_KL(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
                x_star = np.argmax(eta_V_k)/len(eta_V_k)
                eta_max = np.max(eta_V_k)
                print('Certificate argmax = ' + str(eta_max))
                if x_star_old == x_star and eta_max_old == eta_max: 
                    if verbose:
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
                energy_vector[k:] = KLTV_cost_funct(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
                if verbose:
                    print(f'* Energy : {energy_vector[k]:.3f}')
                    print("\n---- Stopping criterion: based on the certificate ----")
                return (measure_k, energy_vector[:k])

            # if <1 at later iterations -> SLIDING
            ####### SLIDING ########
            else:
                measure_k = sliding_step(measure_k)
                # COMPUTE THE CERTIFICATE
                eta_V_k = etak_KL(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
                x_star = np.argmax(eta_V_k)/len(eta_V_k)
                eta_max = np.max(eta_V_k)
                print('Certificate argmax = ' + str(eta_max))
                if eta_max < 1: 
                    energy_vector[k:] = KLTV_cost_funct(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
                    if verbose:
                        print(f'* Energy : {energy_vector[k]:.3f}')
                        print("\n---- Stopping criterion: based on the certificate ----")
                    return (measure_k, energy_vector[:k])
                if x_star_old == x_star and eta_max_old == eta_max: 
                    if verbose:
                        print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                        print('----For loop stopped to avoid infinity loop----')
                    return(measure_k, energy_vector[:k])
                else: 
                    eta_max_old = eta_max
                    x_star_old = x_star

        # reconstruction at iteration k
        a_k = measure_k.a
        x_k = measure_k.x
        Nk = measure_k.N
        energy_vector[k] = KLTV_cost_funct(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
        if verbose:
            print(f'* Energy : {energy_vector[k]:.3f}')

        if plots and measure_k.N>0:
            plt.figure(figsize=(7, 4))
            if gt != 0:
                plt.stem(gt.x, np.array(gt.a), basefmt=" ",
                         linefmt='k--', markerfmt='ko', label='Ground-truth')
            plt.stem(np.array(measure_k.x), np.array(measure_k.a), basefmt=" ",
                     linefmt='r--', markerfmt='ro', label='Recovered')
            plt.plot((0, 1), (0, 0), 'k-', linewidth=1)
            plt.xlim([0, 1])
            plt.legend()
            plt.show()

    if verbose:
        print("\n---- Stopping criterion: End for ----")

    if verbose:
        print('\n' + 'KL-BLASSO with ' + str(k + 1) + ' iterations of Boosted SFW')
        print('* Regularisation parameter Lambda = : ' + str(par_reg))
        print('Reconstruction ')
        print('* positions: ' + str(np.round(measure_k.x, 4)))
        print('* amplitudes : ' + str(np.round(measure_k.a, 4)))
        if gt!=0:
            print('Ground truth ')
            print('* positions: ' + str(np.round(gt.x, 4)))
            print('* amplitudes : ' + str(np.round(gt.a, 4)))

    return (measure_k, energy_vector)
