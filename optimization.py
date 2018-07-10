import numpy as np
from scipy.fftpack import dst, idst, fftfreq, dct, idct
from scipy.special import gamma
from matplotlib import pyplot as plt
from numba import jit, autojit
from solvefokkerplanck_functions import *


def error(alpha_new,quant_arr,pde_new,xlim,alpha_old=None,pde_old=None):
    ''' Inputs: quant_array is the cumulative probability dist for the QRW
        pde_array is the cumulative prob dist for the PDE solution. If the xlim is 
        x times bigger than the (quant_array[1]-1)/2, then the pde_array.shape[1]-1 should be
        nx times bigger than quant_array.shape[1]-1 where n is an int. pde_array.shape[0] should be
        m times bigger than quant_array.shape[0]-1.
        Outputs: [functional value, gradient that comes from the soln].
    '''
    if (alpha_old is None) or (pde_old is None):
        pde_new = construct_pdeCDF(pde_new,4*xlim/(pde_new.shape[0]-1))
        colmn_steps1 = int(np.ceil(pde_new.shape[1]/quant_arr.shape[1]))
        x_stepsize1 = 2*xlim/(pde_new.shape[0]-1)
        row_steps1 = int(1/x_stepsize1)
        steps = int((quant_arr.shape[0]-1)/2)
        newquant_arr = np.concatenate((np.zeros((int(xlim-steps),steps+1)),quant_arr,np.ones((int(xlim-steps),steps+1))),axis=0)
        #row_lim1 = int(steps*row_steps1)
        #middle1 = int((pde_new.shape[0]-1)/2)
        #left_lim1 = middle1-row_lim1
        #right_lim1 = middle1+row_lim1+1
        approx_new = pde_new[::row_steps1,::colmn_steps1]
        functional = .5*np.sum((newquant_arr-approx_new)**2)
        return [functional, None]
    else:
        #construct the pde cumulative array and compute the column steps we need to take
        pde_new = construct_pdeCDF(pde_new,4*xlim/(pde_new.shape[0]-1))
        pde_old = construct_pdeCDF(pde_old,4*xlim/(pde_old.shape[0]-1))
        colmn_steps1 = int(np.ceil(pde_new.shape[1]/quant_arr.shape[1]))
        colmn_steps2 = int(np.ceil(pde_old.shape[1]/quant_arr.shape[1]))
        #compute the x_stepsize and how we need to jump row wise
        x_stepsize1 = 2*xlim/(pde_new.shape[0]-1)
        x_stepsize2 = 2*xlim/(pde_old.shape[0]-1)
        row_steps1 = int(1/x_stepsize1)
        row_steps2 = int(1/x_stepsize2)
        
        #quantum number of steps
        steps = int((quant_arr.shape[0]-1)/2)
        newquant_arr = np.concatenate((np.zeros((int(xlim-steps),steps+1)),quant_arr,np.ones((int(xlim-steps),steps+1))),axis=0)
        #compute how far we need to extend our array from the middle
        #row_lim1 = int(steps*row_steps1)
        #row_lim2 = int(steps*row_steps2)

        #middle1 = int((pde_new.shape[0]-1)/2)
        #middle2 = int((pde_old.shape[0]-1)/2)
        
        #comput the left and right lims and slice our array
        #left_lim1 = middle1-row_lim1
        #right_lim1 = middle1+row_lim1+1
        #left_lim2 = middle2-row_lim2
        #right_lim2 = middle2+row_lim2+1
        approx_new = pde_new[::row_steps1,::colmn_steps1]
        approx_old = pde_old[::row_steps2,::colmn_steps2]
        #soln_gradient
        finite_diff = (approx_new-approx_old)/(alpha_new-alpha_old)
        soln_grad = np.sum(((approx_new-newquant_arr)*finite_diff))/np.size(quant_arr)
        #the 2 norm of the error squared
        functional = .5*np.sum((newquant_arr-approx_new)**2)/np.size(quant_arr)
        return [functional, soln_grad] 

def gradient(alpha_new,soln_grad):
    ''' Inputs: alpha_new, soln_grad are all floats.
        Outputs: a float that is the value of the gradient'''
    alpha_grad = .00001*(2*alpha_new-1)/(((1-alpha_new)*alpha_new)**2)
    return alpha_grad+soln_grad

def initialization_func(log,quant_arr,alpha,x_lim,x_steps,t_lim,t_steps,stri):
    '''Inputs: quant_arr,alpha,x_lim,x_steps,t_lim,t_steps, are all floats or float
               arrays from alpha_optimization. stri is a string that must be old or new
       Outputs: a list that looks like [soln_new, error_new] where soln_new is the solved pde
                and error_new is a float of the error.
       Other notes: Please note that this is used so that we can get an accurate gradient
                    when solving the pde with a refined mesh. This function acts as a general
                    way to do that.'''
    if stri=="old":
        #log.write a messages for the user so they can stay up to date
        log.write("\nCurrent alpha: "+str("{0:.10f}".format(alpha))+"\nMesh Stats: "
              +"dx="+str("{0:.5f}".format(2*x_lim/(x_steps-1)))
              +", dt="+str("{0:.5f}".format(t_lim/t_steps))+"\n")
        #we first calculate
        log.write("\nRUNNING: solve_fokkerplanck("+str(alpha)+","+str(2*x_lim)+","+str(x_steps)+","+str(t_lim)+","+str(t_steps)+")\n")
        soln_new = solve_fokkerplanck(alpha,2*x_lim,x_steps,t_lim,t_steps)
        error_new = error(alpha,quant_arr,soln_new,x_lim)[0]
        return [soln_new,error_new]
    elif stri=="new":
        #log.write a messages for the user so they can stay up to date
        log.write("\nCurrent alpha: "+str("{0:.10f}".format(alpha))+"\nMesh Stats: "
              +"dx="+str("{0:.5f}".format(2*x_lim/(x_steps-1)))
              +", dt="+str("{0:.5f}".format(t_lim/t_steps)))
        #we first calculate
        log.write("\nRUNNING: solve_fokkerplanck("+str(alpha)+","+str(2*x_lim)+","+str(x_steps)+","+str(t_lim)+","+str(t_steps)+")")
        soln_new = solve_fokkerplanck(alpha,2*x_lim,x_steps,t_lim,t_steps)
        
        error_new = error(alpha,quant_arr,soln_new,x_lim)[0]
        return [soln_new,error_new]

def alpha_optimization(log,alpha_init,steps,meshsize=5,learn_rt=.1,tol0=10**(-4)):
    log.write("\nRUNNING: alpha_optimization("+str(log)+","+str(alpha_init)+","+str(steps)+","+str(learn_rt)+","+str(tol0)+")"+"\n")
    alpha_func_list = [[],[],[]]
    #initialize the alpha_new and alpha_old
    alpha_new = alpha_init
    alpha_old = ((1-alpha_init)/10)+alpha_init
    
    #initialize our pde parameters
    x_lim = 200
    x_steps = int(meshsize*10*x_lim)+1
    t_lim = 2*np.sqrt(2)*steps
    t_steps = int(meshsize*15*steps)
    
    #construct the QRW CDF that we'll use throughout the program
    quant_arr = construct_quantarray(steps)
    grad = tol0+1
    refine_thresh = tol0
    try:
        [soln_new, error_new] = initialization_func(log,quant_arr,alpha_old,x_lim,x_steps,t_lim,t_steps,"old")

    except ConvergenceException as e:
        #log.write the message, refine the mesh in time and log.write the new t_steps
        log.write(str(e))
        #we want to refine the mesh 
        t_steps = 2*t_steps
        log.write("\nNew t_steps: "+str(t_steps)+"\n")
                
        [soln_new, error_new] = initialization_func(log,quant_arr,alpha_old,x_lim,x_steps,t_lim,t_steps,"old")

    while np.abs(grad)>tol0:
        excep = True
        while excep:
            try:
                #update old values to what the old values were
                soln_old = soln_new
                error_old = error_new
                
                #log.write a messages for the user so they can stay up to date
                log.write("\nCurrent alpha: "+str("{0:.10f}".format(alpha_new))+"\nMesh Stats: "
                      +"dx="+str("{0:.5f}".format(2*x_lim/(x_steps-1)))
                      +", dt="+str("{0:.5f}".format(t_lim/t_steps)))
                #solve the pde based on the latest new alpha
                #if we get an exception, it should be here
                log.write("\nRUNNING: solve_fokkerplanck("+str(alpha_new)+","+str(2*x_lim)+","+str(x_steps)+","+str(t_lim)+","+str(t_steps)+")")
                soln_new = solve_fokkerplanck(alpha_new,2*x_lim,x_steps,t_lim,t_steps)
                
                
                [error_new, soln_grad] = error(alpha_new,quant_arr,soln_new,x_lim,alpha_old=alpha_old,pde_old=soln_old)
                
                #compute the gradient
                grad = gradient(alpha_new,soln_grad)
                log.write("\nGradient: "+str(grad))
                log.write("\nFunctional: "+str(error_new)+"\n")
                
                #after computing the gradient, check to see whether it is less than tol0
                #if so log.write the final alpha and return the list
                if np.abs(grad)<tol0:
                    log.write("---------------------------------------------------FINAL ALPHA: "+str("{0:.10f}".format(alpha_new))+"---------------------------------------------------\n")
                    return alpha_func_list
                
                #update the old to the next alpha
                alpha_old = alpha_new
                
                #update alpha_new based on the gradient descent method
                alpha_new = alpha_new-grad*learn_rt
                alpha_func_list[0].append(alpha_new)
                alpha_func_list[1].append(error_new)
                alpha_func_list[2].append(grad)
                
                #if we get close to the min
                if np.abs(grad)-tol0<refine_thresh:
                    '''#refine the mesh
                    x_steps = 2*x_steps-1
                    t_steps = 4*t_steps
                    #update the learning rate and threshold'''
                    #learn_rt = learn_rt/2
                    #refine_thresh=refine_thresh/10
                    log.write("\nGetting closer to the minimum. Decreasing learning rate.")
                    '''log.write("\nNew x_steps: "+str(x_steps)+", New t_steps: "+str(t_steps))
                    log.write("\nNew learning rate: "+str("{0:.5f}".format(learn_rt)))
                    #redo the initialization with the refined mesh so we dont get weird gradients
                    [soln_new, error_new] = initialization_func(log,quant_arr,alpha_new,x_lim,x_steps,t_lim,t_steps,"new")'''
                #if there were no issues, we set excep to false so we exit the inner while loop
                excep = False
                
            #if there was a ConvergenceException, do what is in the block
            #the idea behind making this type of exception is so we can get away with coarse
            #meshes to speed up our program, but refine it when we need to
            #any other type of error will be handled elsewhere which is what we want
            except ConvergenceException as e:
                #log.write the message, refine the mesh in time and log.write the new t_steps
                log.write(str(e))
                #we want to refine the mesh 
                t_steps = 2*t_steps
                log.write("New t_steps: "+str(t_steps)+"\n")
                
                #solve the pde with the old alpha here, and we will then try computing everything again in the try block
                [soln_new, error_new] = initialization_func(log,quant_arr,alpha_old,x_lim,x_steps,t_lim,t_steps,"old")
                
    log.write("----------FINAL ALPHA: "+str("{0:.10f}".format(alpha_new))+"----------"+"\n")
    return alpha_func_list