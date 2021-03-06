from optimization import *
from solvefokkerplanck_functions import *
import numpy as np
import matplotlib.pyplot as plt
import time



def alpha_optimization_master(filename="optimization_log.txt",step_attempts=[50],ic_attempts=[.5],learn_rate=.1,tol=10**(-4)):

    #start time
    t0 = time.time()

    #create and open a new txt file
    log = open(filename,"w+")
    log.write("RUNNING: alpha_optimization_master(filename="+str(filename)+
    ",step_attempts="+str(step_attempts)+",ic_attempts="+str(ic_attempts)+
    ",learn_rate="+str(learn_rate)+",tol="+str(tol)+")")


    #looping over different steps values
    for steps in step_attempts:
        log.write('\n\n---------------------------------------------------STEPS: '+str(steps)+"---------------------------------------------------\n")
        #loop over different initial alpha
        for alpha_init in ic_attempts:

            #log the initial alpha and try to optimize alpha
            log.write('\nalpha_init: '+str(alpha_init))
            try:
                #optimize for alpha and store the array of errors and alphas
                start = time.time()
                alpha_guess = alpha_optimization(log,alpha_init,steps,meshsize=1,learn_rt=2*learn_rate,tol0=tol)[0][-1]
                results_list = np.asarray(alpha_optimization(log,alpha_guess,steps,meshsize=2,learn_rt=learn_rate,tol0=tol))
                stop = time.time()
                log.write("Optimization time in seconds: "+str(stop-start))
                results_filename = filename[:-8]+"_"+str(steps)+"steps_"+str(alpha_init)+"ic"
                np.save(results_filename,results_list)

                #plotting the error vs alpha and saving the figure
                plt.figure(figsize=(10,7))
                plt.plot(results_list[0,:],results_list[1,:],'.')
                plt.xlabel('alpha')
                plt.ylabel('Error')
                plt.title('Error vs. alpha at '+str(steps)+' steps and alpha_init='+str(alpha_init))
                plt.savefig(results_filename+'errorvalpha.pdf', format='pdf')
                plt.close()

                plt.figure(figsize=(10,7))
                plt.plot(results_list[1,:],'.')
                plt.xlabel('Iteration')
                plt.ylabel('Error')
                plt.title('Error vs. Iteration at '+str(steps)+' steps and alpha_init='+str(alpha_init))
                plt.savefig(results_filename+'_errorviteration.pdf', format='pdf')
                plt.close()

                plt.figure(figsize=(10,7))
                plt.plot(np.abs(results_list[2,:]),'.')
                plt.xlabel('Iteration')
                plt.ylabel('Gradient')
                plt.title('Gradient vs. Iteration at '+str(steps)+' steps and alpha_init='+str(alpha_init))
                plt.savefig(results_filename+'_gradientviteration.pdf', format='pdf')
                plt.close()
            
            #if we get an exception in our program that is not handled in the optimization process
            #then we handle it here by continuing with the loop
            except Exception as e:
                log.write(str(e)+"\n")
                continue

    #get final time and log.write how long all this took
    t1 = time.time()
    log.write("\nTotal time in seconds: "+str(t1-t0))
    log.close()


#trial run
#alpha_optimization_master(filename="optimization5_log.txt",step_attempts=np.arange(10,40,10),ic_attempts=[.5],learn_rate=.5,tol=10**(-4))