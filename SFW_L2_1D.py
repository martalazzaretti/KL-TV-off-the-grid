from measure_1d import*
import scipy
import matplotlib.pyplot as plt

def SFW_L2_1D(X, acquis, bg, sigma, ker='gaussian', m_0=0, par_reg=1e-5, nIter=5, sliding = 1, pruning = 1, verbose = False, plots = False, gt = 0):
    
    # initialisation
    if m_0==0:
        m_0 = Measure([],[])    # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure(a_k, x_k)
    energy_vector = np.zeros(nIter)
    x_star_old = []
    eta_max_old = []
    
    for k in range(nIter):
        if verbose:
            print('\n' + 'Iteration n. ' + str(k+1))
        
        # step 1: compute the position for the new spike
        eta_V_k = etak_L2(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
        N_ech_eta = len(eta_V_k)
        x_star_index = np.argmax(np.abs(eta_V_k))
        x_star = x_star_index / N_ech_eta
        eta_max = eta_V_k[x_star_index]
        
        if x_star_old == x_star and eta_max_old == eta_max: 
            if verbose:
                print('\n----Stopping criterion: max and argmax value of the certificate not changed----')
                print('----For loop stopped to avoid infinity loop----')
            return(measure_k, energy_vector[:k])
        else: 
            eta_max_old = eta_max
            x_star_old = x_star
            
        if verbose:
            print(f'New position x^* = {x_star} max value of certificate {np.round(eta_max, 4)}')
        
        # Check on the max value of the certificate: if <1 stop
        if eta_max < 1:
            energy_vector[k:] = L2TV_cost_funct(measure_k, X, acquis, bg, par_reg, sigma, ker=ker)
            if verbose:
                print(f'* Energy : {energy_vector[k]:.3f}')
                print("\n---- Stopping criterion: based on the certificate ----")
            return(measure_k, energy_vector[:k])
        else:
            x_k_demi = x_k + [x_star] # I add the new position to the reconstruction
            if verbose: 
                
                print('Measure_k_demi : ')
                print('* x_k_demi : ' + str(np.round(x_k_demi, 4)))
            
            # define the lasso functional at each step: dimensions change! x_k_demi is bigger at each step            
            def lasso(a):
                aus = phi_vector(a,x_k_demi,X,sigma,ker=ker)+bg
                fidelity = 0.5*np.linalg.norm(acquis - aus,2)**2
                penalty = par_reg*np.linalg.norm(a, 1)
                return(fidelity + penalty) 
            def grad_lasso(a):
                grad = phi_vector(a,x_k_demi,X,sigma,ker=ker)+bg
                grad = phi_amplitude_adjoint(grad-acquis, x_k_demi, X, sigma, ker='gaussian')+par_reg*np.sign(a)
                return grad
            # solve the lasso minimisation problem 
            init_guess = a_k+[0]
            res = scipy.optimize.minimize(lasso, init_guess,method='L-BFGS-B')
            a_k_demi = res.x.tolist()
            if verbose: 
                print('* a_k_demi : ' + str(np.round(a_k_demi, 4)))
            
            measure_k_demi = Measure(a_k_demi,x_k_demi)
            
            
            ####### SLIDING ########
            if sliding == 1:
                initial_guess = a_k_demi + x_k_demi # amplitude a che usiamo come inizializzazione sono nella prima metà della
                # lista, posizioni x nella seconda metà
                        
                # define the ''lasso double'' functional: it depends both on a and on x
                def lasso_double(params):
                    a = params[:int(len(params)/2)]
                    x = params[int(len(params)/2):]
                    aus = phi_vector(a,x,X,sigma,ker=ker)+bg
                    fidelity = 0.5*np.linalg.norm(acquis - aus,2)**2
                    penalty = par_reg*np.linalg.norm(a, 1)
                    return(fidelity + penalty)
                # solve the double optimisation problem
                res = scipy.optimize.minimize(lasso_double, initial_guess,
                                              method='L-BFGS-B', 
#                                               bounds=bnds_new,
                                              options={'disp': True})
                a_k_plus = (res.x[:int(len(res.x)/2)]).tolist()
                x_k_plus = (res.x[int(len(res.x)/2):]).tolist()
                # compute the corresponding ''acquired measure''
                measure_k = Measure(a_k_plus, x_k_plus)
                
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
                    
                
            # reconstruction at iteration k    
            a_k = measure_k.a
            x_k = measure_k.x
            Nk = measure_k.N
            energy_vector[k] = L2TV_cost_funct(measure_k,X, acquis, bg, par_reg, sigma, ker=ker)
            if verbose: 
                print(f'* Energy : {energy_vector[k]:.3f}')
                
            if plots and measure_k.N>0: 
                plt.figure(figsize=(7,4))
                if gt != 0:
                    plt.stem(gt.x, np.array(gt.a), basefmt=" ",
                         linefmt='k--', markerfmt='ko', label='Ground-truth')
                plt.stem(np.array(measure_k.x), np.array(measure_k.a), basefmt=" ",
                     linefmt='r--', markerfmt='ro', label='Recovered')
                plt.plot((0, 1), (0, 0), 'k-', linewidth=1)
                plt.xlim([0,1])
                plt.legend()
                plt.show()
            
    print("\n---- L2-BLASSO Stopping criterion: Max n. iterations ----")
        
    if verbose:
        print('\n' + 'BLASSO with ' + str(k+1) + ' iterations of SFW')
        print('* Regularisation parameter Lambda = : ' + str(par_reg))
        print('Reconstruction ')
        print('* positions: ' + str(np.round(measure_k.x, 4)))
        print('* amplitudes : ' + str(np.round(measure_k.a, 4)))
        if gt != 0:
            print('Ground truth ')
            print('* positions: ' + str(np.round(gt.x, 4)))
            print('* amplitudes : ' + str(np.round(gt.a, 4)))
        
    return(measure_k, energy_vector)