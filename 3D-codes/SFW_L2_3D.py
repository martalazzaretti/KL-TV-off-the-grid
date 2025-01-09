import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from measure_3d import*




def SFW_L2_3D(y, bg, sig_x, sig_y, sig_z, X ,Y , Z,  N_xy, N_z, pixel_size_xy, pixel_size_z, par_reg=1e-5, nIter=5, m_0=0):
    '''y acquisition et nIter number of iterations'''
    if m_0==0:
        m_0 = Measure3D(np.empty((0,0)), np.empty((0,0)))    # empty discrete measure
    Nk=m_0.N      # discrete measure size
    a_k = m_0.a
    x_k = m_0.x
    measure_k = Measure3D(a_k, x_k)
    N_xy_y = len(y)                   # discrete measure size
    
    energy_k = np.zeros(nIter)
    for k in range(nIter):
        
        print('\n' + 'Step number ' + str(k+1))
        start_time = time.process_time()
        # compute the certificate
        eta_V_k = etak_L2(measure_k, y, bg, X, Y, Z, sig_x, sig_y, sig_z, par_reg)
        # find argmax --> potential position of the new spike
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None), eta_V_k.shape)
        # compute the position
        print('-Insertion step: compute new position')
        x_star = [(x_star_index[1]-N_xy/2)*pixel_size_xy, (x_star_index[0]-N_xy/2)*pixel_size_xy, (x_star_index[2]-N_z/2)*pixel_size_z]
        print(f'* x^* index {x_star} max = {np.round(eta_V_k[x_star_index], 3)}')
        end_time = time.process_time()
        print("Computation of certificate and insertion of new spike: process time", end_time - start_time)
        
#         fig=plt.figure(figsize=(12, 3))
#         for i in range(3): 
#             plt.subplot(1,3,i+1)
#             plt.imshow(np.amax(eta_V_k[:,:,:],axis=i))
#             if i == 0:
#                 plt.scatter(x_star[2]/pixel_size_z+N_z/2, x_star[0]/pixel_size_xy+N_xy/2, c='white')
#             elif i == 1:
#                 plt.scatter(x_star[2]/pixel_size_z+N_z/2, x_star[1]/pixel_size_xy+N_xy/2, c='white')        
#             else: 
#                 plt.scatter(x_star[0]/pixel_size_xy+N_xy/2, x_star[1]/pixel_size_xy+N_xy/2, c='white')
#         fig.suptitle('Certificate and its argmax')
#         plt.show()

        # Stopping criterion (step 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            energy_k[k] = L2TV_cost_funct(measure_k, y, bg,  X, Y, Z, sig_x, sig_y, sig_z, par_reg)
            print(f'* Energy: {energy_k[k]:.3f}')
            print("\n\n---- Halting condition ----")
            return(measure_k, energy_k)
        else:
            start_time = time.process_time()
            # adding new spike position to the vector of positions
            if x_k.size == 0:
                x_k_demi = np.vstack([x_star])
                a_k_demi = 0.1
            else:
                x_k_demi = np.vstack([x_k, x_star])
                a_k_demi = np.concatenate((a_k, 0.1),axis=None)

            # Solving LASSO (step 7)
            def lasso(a):
                fidelity = 0.5*my3dnorm(y - phi_vector(a,x_k_demi,X,Y,Z,sig_x, sig_y, sig_z))**2
                penalty = par_reg*np.linalg.norm(a, 1)
                return(fidelity + penalty)
            res = scipy.optimize.minimize(lasso, a_k_demi, method='L-BFGS-B', options={'disp':False})
            a_k_demi = res.x
            measure_k_demi = Measure3D() # empty measure 
            measure_k_demi += Measure3D(a_k_demi,x_k_demi)
            print('-Estimation of the amplitudes after the insertion of the new spike position')
            print('* a_k_demi : ' + str(np.round(a_k_demi, 2))) 
            print('* x_k_demi : ' + str(np.round(x_k_demi, 2)))
            end_time = time.process_time()
            print("Estimation of amplitudes: process time", end_time - start_time)
            
#             fig=plt.figure(figsize=(12, 3))
#             for i in range(3): 
#                 plt.subplot(1,3,i+1)
#                 plt.imshow(np.amax(y[:,:,:],axis=i))
#                 if i == 0:
#                     plt.scatter(m_ax0.x[:,2]/pixel_size_z+N_z/2, m_ax0.x[:,0]/pixel_size_xy+N_xy/2, c='green')
#                     plt.scatter(measure_k_demi.x[:,2]/pixel_size_z+N_z/2, measure_k_demi.x[:,0]/pixel_size_xy+N_xy/2, c='red')
#                 elif i == 1:
#                     plt.scatter(m_ax0.x[:,2]/pixel_size_z+N_z/2, m_ax0.x[:,1]/pixel_size_xy+N_xy/2, c='green')
#                     plt.scatter(measure_k_demi.x[:,2]/pixel_size_z+N_z/2, measure_k_demi.x[:,1]/pixel_size_xy+N_xy/2, c='red')
#                 else: 
#                     plt.scatter(m_ax0.x[:,0]/pixel_size_xy+N_xy/2, m_ax0.x[:,1]/pixel_size_xy+N_xy/2, c='green')
#                     plt.scatter(measure_k_demi.x[:,0]/pixel_size_xy+N_xy/2,measure_k_demi.x[:,1]/pixel_size_xy+N_xy/2, c='red')
#             fig.suptitle('Insertion step related to argmax of certificate')
#             plt.show()

            # Solving non-convex double LASSO (step 8)
            start_time = time.process_time()
            def lasso_double(params):
                a_p = params[:int(len(params)/4)] 
                x_p = params[int(len(params)/4):]
                x_p = x_p.reshape((len(a_p), 3))
                fidelity = 0.5*my3dnorm(y - phi_vector(a_p,x_p,X,Y,Z,sig_x, sig_y, sig_z))**2
                penalty = par_reg*np.linalg.norm(a_p, 1)
                return(fidelity + penalty)

            def grad_lasso_double(params):
                a_p = params[:int(len(params)/4)]
                x_p = params[int(len(params)/4):]
                x_p = x_p.reshape((len(a_p), 3))
                N = len(a_p)
                partial_a = N*[0]
                partial_x = 3*N*[0]
                residual = y - phi_vector(a_p,x_p,X,Y,Z,sig_x, sig_y, sig_z)-bg
                for i in range(N):
                    integ = np.sum(residual*gaussian_3D(X - x_p[i, 0], Y - x_p[i, 1], Z - x_p[i,2],sig_x, sig_y, sig_z))
                    partial_a[i] = par_reg - integ/N_xy_y

                    grad_gauss_x = grad_x_gaussian_3D(X - x_p[i, 0], Y - x_p[i, 1], Z - x_p[i,2], X - x_p[i, 0],sig_x, sig_y, sig_z)
                    integ_x = np.sum(residual*grad_gauss_x) 
                    partial_x[3*i] = a_p[i] * integ_x/(3*N_xy_y)
                    
                    grad_gauss_y = grad_y_gaussian_3D(X - x_p[i, 0], Y - x_p[i, 1], Z - x_p[i,2], Y - x_p[i, 1],sig_x, sig_y, sig_z)
                    integ_y = np.sum(residual*grad_gauss_y)
                    partial_x[3*i+1] = a_p[i] * integ_y / (3*N_xy_y)
                    
                    grad_gauss_z = grad_z_gaussian_3D(X - x_p[i, 0], Y - x_p[i, 1], Z - x_p[i,2], Z - x_p[i, 2],sig_x, sig_y, sig_z)
                    integ_z = np.sum(residual*grad_gauss_z)
                    partial_x[3*i+2] = a_p[i] * integ_z / (3*N_xy_y)

                return(partial_a + partial_x)

            # solve with scipy.optimize
            initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))                  
            res = scipy.optimize.minimize(lasso_double, initial_guess,
                                          method='L-BFGS-B',
                                          #jac=grad_lasso_double,
                                          options={'disp': False})
            a_k_plus = (res.x[:int(len(res.x)/4)])
            x_k_plus = (res.x[int(len(res.x)/4):]).reshape((len(a_k_plus), 3))
            print('-Sliding step: restimation of amplitudes and positions with a non-convex minimization step')
            print('* a_k_plus : ' + str(np.round(a_k_plus, 2))) 
            print('* x_k_plus : ' + str(np.round(x_k_plus, 2)))
            
            # Update of parameters a and x and pruning of zero diracs
            measure_k = Measure3D(a_k_plus, x_k_plus)
            end_time = time.process_time()
            print("Sliding step: process time", end_time - start_time)
            
#             measure_k = Measure3D(a_k_demi, x_k_demi) # no sliding

            start_time = time.process_time()
            measure_k = measure_k.prune(tol=1e-3)
            a_k = measure_k.a
            x_k = measure_k.x
            Nk = measure_k.N
            print('-Pruning')
            print(f'* {Nk} Diracs')
            print('* a_k : ' + str(np.round(a_k, 2))) 
            print('* x_k : ' + str(np.round(x_k, 2)))
            energy_k[k] = L2TV_cost_funct(measure_k, y, bg,  X, Y, Z, sig_x, sig_y, sig_z, par_reg)
            print(f'* Energy: {energy_k[k]:.3f}')
            end_time = time.process_time()
            print("Pruning: process time", end_time - start_time)
#             fig=plt.figure(figsize=(12, 3))
#             for i in range(3): 
#                 plt.subplot(1,3,i+1)
#                 plt.imshow(np.amax(y[:,:,:],axis=i))
#                 if i == 0:
#                     plt.scatter(m_ax0.x[:,2]/pixel_size_z+N_z/2, m_ax0.x[:,0]/pixel_size_xy+N_xy/2, c='green')
#                     plt.scatter(measure_k.x[:,2]/pixel_size_z+N_z/2, measure_k.x[:,0]/pixel_size_xy+N_xy/2, c='red')
#                 elif i == 1:
#                     plt.scatter(m_ax0.x[:,2]/pixel_size_z+N_z/2, m_ax0.x[:,1]/pixel_size_xy+N_xy/2, c='green')
#                     plt.scatter(measure_k.x[:,2]/pixel_size_z+N_z/2, measure_k.x[:,1]/pixel_size_xy+N_xy/2, c='red')
#                 else: 
#                     plt.scatter(m_ax0.x[:,0]/pixel_size_xy+N_xy/2, m_ax0.x[:,1]/pixel_size_xy+N_xy/2, c='green')
#                     plt.scatter(measure_k.x[:,0]/pixel_size_xy+N_xy/2,measure_k.x[:,1]/pixel_size_xy+N_xy/2, c='red')
#             fig.suptitle('Sliding step: re-evaluation of positions and amplitudes of spikes')
#             plt.show()
            
                        
    print("\n\n---- End of the loop ----")
    return(measure_k, energy_k)
