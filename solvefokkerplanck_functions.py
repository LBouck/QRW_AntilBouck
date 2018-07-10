import numpy as np
from scipy.fftpack import dst, idst, fftfreq, dct, idct
from scipy.special import gamma
from numba import autojit, jit

#ConvergenceException is an exception class that will be raised specifically if our FPI is not converging
#we'll handle it later by refining the mesh
class ConvergenceException(Exception):
    #constructor method
    def __init__(self, value):
        self.value = value
    #string method
    def __str__(self):
        return repr(self.value)


        

def dirac(x):
    '''Takes an array symmetric at x=0 and returns the dirac delta approximation of it at 0'''
    ans=0*x
    #if we have an even length array the middle two elements are taken to be a height
    #so that the integral is 1
    if len(x)%2==0:
        ans[int(len(ans)/2)-1]=1/(3*x[int(len(ans)/2)]-x[int(len(ans)/2)-1])
        ans[int(len(ans)/2)]=1/(3*x[int(len(ans)/2)]-x[int(len(ans)/2)-1])
    #if the array is odd, then we take the middle three elements with half the height to make the integral 1
    else:
        ans[int(len(ans)/2)-1]=1/(3*(x[int(len(ans)/2)]-x[int(len(ans)/2)-1]))
        ans[int(len(ans)/2)]=1/(3*(x[int(len(ans)/2)]-x[int(len(ans)/2)-1]))
        ans[int(len(ans)/2)+1]=1/(3*(x[int(len(ans)/2)]-x[int(len(ans)/2)-1]))
    return ans

def quantumwalk_line(steps,initial=np.array([np.sqrt(.5), -1j*np.sqrt(.5)])):
    """quantumwalk_line takes two inputs, steps is the natural number of time steps you want to 
    take of a Hadamard walk, and initial is a initial superposition of the probability amplitudes
    of the coin state with |0>=[1,0] and |1>=[0,1]. The output will be a matrix of size 2Xsteps
    where each coulumn is the super position of the coin basis states corresponding with the
    position basis state."""
    
    #create initial array
    #make sure the initial array type is complex or else numpy will 
    #keep only the real parts
    mat=np.zeros((2,2*steps+1),dtype=complex)
    mat[0,steps]=initial[0]
    mat[1,steps]=initial[1]
    
    #coin operator
    H=np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2),-1/np.sqrt(2)]])
    
    for step in range(1,steps+1):
        #coin operator acts on system
        mat=np.dot(H,mat)
        #shift operator acts on system
        #first one shifts positions to the left
        for i in range(2*steps):
            mat[1,i]=mat[1,i+1]
        mat[1,2*steps]=0
        #shift positions to the right
        for j in range(2*steps,0,-1):
            mat[0,j]=mat[0,j-1]
        mat[0,0]=0
    return mat

def probabilities(mat):
    '''inputs is a 2Xsteps matrix from quantumwalk_line. This function will output a numpy 
    1Xsteps array of the probability of the walker being at a certain position'''
    mat1=np.square(np.absolute(mat))
    mat2=np.zeros((1,len(mat1[0,:])))
    for i in range(len(mat1[0,:])):
        mat2[0,i]=abs(mat1[1,i])+abs(mat1[0,i])
    return mat2

def construct_quantarray(steps):
    ans = np.zeros((2*steps+1,steps+1))
    ans[steps:,:] = np.ones((steps+1,steps+1))
    for i in range(0,steps+1):
        ans[steps-i:steps+i+1,i]=quantcumprob(probabilities(quantumwalk_line(i))[0,:])
    return ans


def dst_forcingfunc(forcing_function,x_leftlim,x_rightlim,x_steps,t_lim,t_steps):
    tau=t_lim/t_steps
    forcingdst_array=np.zeros((x_steps,t_steps+1))
    for i in range(0,t_steps+1):
        forcingdst_array[:,i]=dst(forcing_function(tau*i,np.linspace(x_leftlim,x_rightlim,x_steps)),type=2,norm='ortho')
    return forcingdst_array

'''#autojit really speeds up this function
@autojit
def iterator_func(inputarray,constant,alpha):
    #store the integer that will be looped up too
    k=inputarray.shape[1]-1
    #if our input array was more than one wide
    if k>0:
        #initialize the answer
        answer=np.zeros(inputarray.shape[0])
        #this loop is the implementation of the L1 scheme for a fractional derivative
        for l in range(0,k):
            a=(k+1-l)**(1-alpha)-(k-l)**(1-alpha)
            intermediate=inputarray[:,l+1]-inputarray[:,l]
            answer=answer-intermediate*a
        answer=(inputarray[:,k]+answer)*constant
    else:
        #otherwise we just deal with the first column for the frac deriviative
        answer = np.zeros(inputarray.shape[0])
        answer = constant*inputarray[:,k]
    return answer'''

# #releasethegil haha
@jit(nogil=True)
def iterator_func(inputarray,constant, alpha):
    ''' Inputs: input array comes from the pde solver and is every transformed 
        solution value before the current time iteration. constant is a float from the pde 
        and the float alpha is the order of differentiation.

        Outputs: outputs what part of the fractional derivative operator would do except for the last
        part of the summation. This comes from our method for the pde.'''

    #take k to be the columns-1
    k=inputarray.shape[1]-1
    #if k isnt 0
    if k>0:
        answer=np.zeros(inputarray.shape[0])
        #loop over rows and then l loops over our scheme for the fractional derivative
        for j in range(0,inputarray.shape[0]):
            for l in range(0,k):
                #this constant a comes from our scheme of the fractional derivative
                a=(k+1-l)**(1-alpha)-(k-l)**(1-alpha)
                #finite difference and contributing it to answer
                intermediate=inputarray[j,l+1]-inputarray[j,l]
                answer[j]=answer[j]-intermediate*a
            answer[j]=(inputarray[j,k]+answer[j])*constant
    #otherwise
    else:
        answer=np.zeros(inputarray.shape[0])
        #just loop over the rows and multiply by the last element
        for j in range(0,inputarray.shape[0]):
            answer[j] = constant*inputarray[j,k]
    return answer

def solve_fokkerplanck(alpha,x_lim,x_steps,t_lim,t_steps,initial_conditions=dirac):
    print("\nRUNNING: solve_fokkerplanck("+str(alpha)+","+str(x_lim)+","+str(x_steps)+","+str(t_lim)+","+str(t_steps)+")")
    #initializing the solution and dst of the solution
    solution=np.zeros((x_steps,t_steps+1))
    solution_hat=np.zeros((x_steps,t_steps+1))
    #time step
    tau=t_lim/t_steps
    #spatial mesh
    xvals=np.linspace(-x_lim,x_lim,x_steps)
    #print(x_vals)
    #hyperbolic tangent values on the mesh
    tanhvals=np.tanh(xvals)

    #first column are the initial conditions
    solution[:,0]=initial_conditions(xvals)
    solution_hat[:,0]=dst(solution[:,0],norm='ortho')

    #constant that is used in the implementation, we call the gamma function
    constant=1/(gamma(2-alpha)*(tau**alpha))

    #computing the array of frequency variables
    omega_array=np.zeros((x_steps,1))
    N=len(xvals)
    L=2*x_lim
    omega_array=np.arange(1,N+1)*np.pi/L
    #note that .5 is the diffusion coefficient
    omega_array=constant*np.ones(x_steps)+.5*(omega_array**2)
    omega_array=1/omega_array
    
    #prescribed tolerance for the fixed point iteration in the while loop
    tol=10**(-14)
    
    #iterating over time steps
    for k in range(0,t_steps):
        #reset error to be big
        err=1
        #reset counter
        count=0
        #first guess of next u is our previous u
        u_old=solution[:,k]
        #while the error is still bigger than we want
        while err>tol:
            #counter to prevent infinite loop. We raise an ConvergenceException to be handled in the optimization function
            count=count+1
            if count>500:
                raise ConvergenceException("FPI not converging. Backward Error is "+str(err)+" after "+str(count-1)+" steps.")
            #take fourier trans of integral using u from previous step
            integ_hat=np.zeros(N)
            integ_hat[0:N-1]=-np.pi*np.arange(1,N)*dct(tanhvals*u_old,norm='ortho')[1:]/L
            #use iterator to get approximation of fractional derivative
            frac_deriv=iterator_func(solution_hat[:,0:k+1],constant,alpha)
            #get approximation for u_hat and take inverse fourier transform
            u_hat=omega_array*(-integ_hat+frac_deriv)
            u_new=np.real(idst(u_hat,norm='ortho'))
            #new error val and use the new approx for next guess
            err=np.abs(np.max((u_new-u_old)))
            u_old=u_new
        #notify user how progress is going
        print('\r'+str("{0:.1f}".format((k/t_steps)*100))+'%',end='')
        #once our guess is good enough, we'll use it for our solution and we'll also keep track of the dst of the soln
        solution[:,k+1]=u_new
        solution_hat[:,k+1]=u_hat
    #return the final array
    return solution


def quantcumprob(array):
    '''Input: array from the quantum random walk, which is a probability mass function.
       Output: array that is the cumulative probability distribution of the QRW'''
    answer=np.zeros(len(array))
    answer[0]=array[0]
    for i in range(1,len(answer)):
        answer[i]=array[i]+answer[i-1]
    return answer


def contcumprob(array,stepsize):
    '''Input: array from solving the PDE and stepsize which is a float
        Output: Uses trapizoidal rule to return a cumulative probability distribution
        of the PDE solution at the time value'''
    answer=np.zeros(len(array))
    for i in range(1,len(answer)):
        answer[i]=stepsize*(array[i]+array[i-1])/2+answer[i-1]
    return answer


def construct_pdeCDF(soln,x_stepsize):
    ''' Inputs: soln array from the pde and x_stepsize which is a float
        Outputs: returns an array that provides the CDF of the soln at
        each time value, which are the columns of the array.'''
    #initialize array
    ans = np.zeros((soln.shape[0],soln.shape[1]))
    #loop over columns
    for i in range(0,soln.shape[1]):
        #compute CDF using contcumprob
        ans[:,i] = contcumprob(soln[:,i],x_stepsize)
    return ans