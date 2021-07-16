"""Manlin Chawla 01205586"""
"""Final project, part 2"""
import numpy as np
import matplotlib.pyplot as plt
import time
from m1 import bmodel as bm #assumes p2.f90 has been compiled with: f2py3 -c p2_dev.f90 -m m1

def simulate_jacobi(n,input_num=(10000,1e-8),input_mod=(1,1,1,2,1.5),display=False):
    """ Solve contamination model equations with
        jacobi iteration.
        Input:
            input_num: 2-element tuple containing kmax (max number of iterations
                        and tol (convergence test parameter)
            input_mod: 5-element tuple containing g,k_bc,s0,r0,t0 --
                        g: bacteria death rate
                        k_bc: r=1 boundary condition parameter
                        s0,r0,t0: source function parameters
            display: if True, a contour plot showing the final concetration field is generated
        Output:
            C,deltac: Final concentration field and |max change in C| each iteration
    """
    #Set model parameters------

    kmax,tol = input_num
    g,k_bc,s0,r0,t0 = input_mod
    #-------------------------------------------
    #Set Numerical parameters
    Del = np.pi/(n+1)
    r = np.linspace(1,1+np.pi,n+2)
    t = np.linspace(0,np.pi,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    #Factors used in update equation
    rinv2 = 1.0/(rg*rg)
    fac = 1.0/(2 + 2*rinv2+Del*Del*g)
    facp = (1+0.5*Del/rg)*fac
    facm = (1-0.5*Del/rg)*fac
    fac2 = fac*rinv2

    #set initial condition/boundary conditions
    C = (np.sin(k_bc*tg)**2)*(np.pi+1.-rg)/np.pi

    #set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*np.exp(-20.*((rg-r0)**2+(tg-t0)**2))*(Del**2)*fac

    deltac = []
    Cnew = C.copy()

    #Jacobi iteration
    for k in range(kmax):
        #Compute Cnew
        Cnew[1:-1,1:-1] = Sdel2[1:-1,1:-1] + C[2:,1:-1]*facp[1:-1,1:-1] + C[:-2,1:-1]*facm[1:-1,1:-1] + (C[1:-1,:-2] + C[1:-1,2:])*fac2[1:-1,1:-1] #Jacobi update
        #Compute delta_p
        deltac += [np.max(np.abs(C-Cnew))]
        C[1:-1,1:-1] = Cnew[1:-1,1:-1]
        if k%1000==0: print("k,dcmax:",k,deltac[k])
        #check for convergence
        if deltac[k]<tol:
            print("Converged,k=%d,dc_max=%28.16f " %(k,deltac[k]))
            break

    deltac = deltac[:k+1]

    if display:
        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Final concentration field')

    return C,deltac

def simulate(n,input_num=(10000,1e-8),input_mod=(1,1,1,2,1.5),display=False):
    """ Solve contamination model equations with
        OSI method, input/output same as in simulate_jacobi above
    """
    #Set model parameters------
    kmax,tol = input_num
    g,k_bc,s0,r0,t0 = input_mod

    #Set Numerical parameters
    Del = np.pi/(n+1)
    r = np.linspace(1,1+np.pi,n+2)
    t = np.linspace(0,np.pi,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    #Factors used in update equation
    rinv2 = 1.0/(rg*rg)
    fac = 1.0/(2 + 2*rinv2+Del*Del*g)
    facp = (1+0.5*Del/rg)*fac
    facm = (1-0.5*Del/rg)*fac
    fac2 = fac*rinv2

    #set initial condition/boundary conditions
    C = (np.sin(k_bc*tg)**2)*(np.pi+1.-rg)/np.pi

    #set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*np.exp(-20.*((rg-r0)**2+(tg-t0)**2))*(Del**2)*fac

    #Preallocate empty vector to add maximum change in conentration for each iteration
    deltac = []

    #Intialize Cnew before loop begins, need copy to avoid both C and Cnew binding
    Cnew = C.copy()

    #OSI Iteration
    for k in range(kmax):
        #Compute Cnew, progress through the matrix element-by-element one row at a time
        for i in np.linspace(1,n,n).astype('int'):
            for j in np.linspace(1,n,n).astype('int'):
                Cnew[i,j]=0.5*(-C[i,j]+3*(facm[i,j]*Cnew[i-1,j]+fac2[i,j]*Cnew[i,j-1]+fac2[i,j]*C[i,j+1]+facp[i,j]*C[i+1,j]+Sdel2[i,j]))

        #Compute delta_c
        deltac += [np.max(np.abs(C-Cnew))]

        #Update C before next iteration starts
        C[1:-1,1:-1] = Cnew[1:-1,1:-1]

        #Print statement at every 1000 iterations
        if k%1000==0: print("k,dcmax:",k,deltac[k])

        #Check for convergence
        #Iterations terminate if max change falls below tolerance
        if deltac[k]<tol:
            print("Converged,k=%d,dc_max=%28.16f " %(k,deltac[k]))
            break

    #Appends recent deltac to the vector
    deltac = deltac[:k+1]

    #Plot
    if display==True:
        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Final concentration field')

    return C,deltac


def performance(fignum,display=False):
    """Analyze performance of simulation codes
    Add input/output variables as needed.
    """

    """Performance(1): This figure is a plot comparing the runtimes between
    the Python and Fortran implementation of the OSI method as n is increased.
    To generate this plot I have varied n (n determines the grid size) from 1 to
    40 and plotted the runtimes at each value of n. The plots shows that as n
    increases, as expected each implementation takes longer to run. For small
    values of n, the Python and Fortran implementations both have very similar
    runtimes. However, as n is increased the runtime for the Python implementation
    rapidly diverges (almost exponentially). The runtimes for the Fortran
    implementation of simulate using OSI method also increases but a much
    shallower slope than the Python implementation. There is only a marginal
    increase in runtime for the Fortran and this is because Fortran is a compiled
    language, using multiple for loops in Fortran does not affect runtime.
    Python is an interpreted language, using multiple loops significantly
    increases runtime

    Performance(2): Following on from the plot from performance(1), this figure
    is a plot comparing the Python and Fortran implementation of the OSI method
    as a ratio.In particular,this plot shows the ratio of the runtimes as n
    increases.  I have varied n from 1 to 50 and plotted the
    Python runtime/Fortran runtime at each value of n. As n increases the ratio
    of the run times increases at a steep slope and the Python function takes
    almost 1000 times longer to run for values of N close to 20. The ratio eventually
    starts to decrease and plateau for larger values of N showing that the
    Python function takes almost 600 times longer to run.

    Performance(3): This figure is a plot comparing the Python and Fortran
    implementations of the Jacobi method as a ratio. I have varied n from 1 to 50
    and plotted the Python runtime/Fortran runtime ratio at each value of n. For
    small values of n the runtime ratio increases steeply but as n increases the
    ratio of the runtimes decreases and starts to level.

    Performance(4): This figure is a plot of n against the final deltac value
    when using the OSI method and when using the Jacobi iteration the method.
    The plot shows that for the OSI method for small values of n the deltac
    (the relative error) fluctuates in the range of 0.2*10^-8 – 1.0*10^-8 but as
    n increases the relative error starts to reach 1.0*10^-8 which is the
    tolerance value before the loop breaks. This mean that as n increases the
    tolerance is reached and the loop breaks before kmax iterations have been
    completed. The plot shows the Jacobi method shows that the relative error
    (value of deltac) is higher for smaller values of n. However,the Jacobi method
    reaches the tolerance for a smaller value of n. I have run this plot for
    larger values of n and around n=90 the values of final value of deltac for
    the Jacobi method become higher than the tolerance and increases rapidly. This
    means that for for large values of n the kmax iterations are reached rather
    than the tolerance. So as n increases, the number of iterations it takes for
    the concentration of bacteria to converge is greater. Intuitively, since
    n controls the grid size, as the grid size increases it takes longer for the bacteria
    to spread from the boundary to the middle of the region.

    Performance(5): This figure is a plot of the values of deltac (the relative error)
    over the iterations of each method. The plot shows that for the OSI method
    the relative error starts quite high at the start of the loop but rapidly
    decreases and reaches the tolerance 1.0*10-8. Once this tolerance is reached
    after approximately 1000 iterations, the loop breaks and the values of deltac
    are 0  (This is the horizontal line part). The plot also shows that for the
    Jacobi method the relative error starts off much lower at the beginning of
    the loop and decreases at a much slower rate. It takes more iterations for the
    tolerance to be reached.

    Performance(6):This figure is a plot of the runtimes for the Jacobi method
    and the OSI method in Fortran. To generate this plot I have varied n from 1
    to 50. The plots shows that n increases, as expected each method takes longer
    to run. For small values of n, the Jacobi and OSI method both have very similar
    runtimes. However, as n is increases the runtime for the Jacobi method
    increases at a faster rate than the OSI method. The runtimes for the Fortran
    implementation of simulate using OSI method also increases but a much shallower
    slope. Looking at the plot for performance(5), as n increases the OSI method
    reaches the tolerance faster than the Jacobi method so the loop breaks earlier
    meaning that the OSI method takes less time to run. This is because the OSI method
    uses terms from Cnew which are updated more recently in the same iteration so
    less iterations are needed before convergence."""

    #Plot 1: Comparison of runtime between Python and Fortran implementation
    #(using OSI method) as n is increased
    if fignum==1:
        #Set values of n (determines size of grid)
        nlen=40
        nvalues=np.linspace(1,nlen,nlen).astype('int')

        #Preallocate for efficiency
        runtime_py=np.zeros(nlen)
        runtime_for=np.zeros(nlen)
        #Repeat times and take minimum to get rid of fluctuations in Fortran runtimes
        timerepeat=np.zeros(25)

        for i,n in enumerate(nvalues):
            print(n)
            #Runtime for Python implementation of simulate
            t0=time.time()
            simulate(n)
            t1=time.time()
            runtime_py[i]=t1-t0

            #Minimum runtime for Fortran implementation of simulate
            for j in range(25):
                t2=time.time()
                bm.simulate(n)
                t3=time.time()
                timerepeat[j]=t3-t2
            runtime_for[i]=min(timerepeat)

        #Plot of figure
        plt.hold(True)
        plt.plot(nvalues,runtime_py,'.',c='r',label='Python implementation')
        plt.plot(nvalues,runtime_for,'.',c='b',label='Fortran implementation')
        plt.xlabel('n')
        plt.ylabel('Runtime (s)')
        plt.title('Manlin Chawla: performance(1) \n Runtime of each implementation as n increases ')
        plt.legend()
        plt.hold(False)

        #Set display==false for plots to generate in if__name==main
        if display==True:
            plt.show()

    #Plot 2: Ratio of runtime between Python and Fortran implementation
    #(using OSI method) as n is increased
    if fignum==2:
        #Set values of n (determines grid size)
        nlen=50
        nvalues=np.linspace(1,nlen,nlen).astype('int')

        #Preallocate for efficiency
        runtimeratio_OSI=np.zeros(nlen)

        #runtimeratio_py=runtime_py/runtime_for
        for i in range(nlen):
            print(i)

            #Time Python implementation using OSI
            t0=time.time()
            simulate(nvalues[i])
            t1=time.time()
            runtimepython=t1-t0

            #Time Fortran implementation using OSI
            t2=time.time()
            bm.simulate(nvalues[i])
            t3=time.time()
            runtimefortran=t3-t2

            #Compute ratio for OSI method
            runtimeratio_OSI[i]=runtimepython/runtimefortran

        #Plot of figure
        plt.hold(True)
        plt.plot(nvalues,runtimeratio_OSI,'.',c='r',label='Python (OSI method)')
        plt.xlabel('n')
        plt.ylabel('Ratio')
        plt.title('Manlin Chawla: performance(2) \n Ratio of runtime (Python Runtime\Fortran Runtime) using OSI, as N increases')
        #plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()


    #Plot 3: Ratio of runtime between Python and Fortran implementation
    #(using Jacobi method) as n is increased
    if fignum==3:
        #Set values of n (determines grid size)
        nlen=50
        nvalues=np.linspace(1,nlen,nlen).astype('int')

        #Preallocate for efficiency
        runtimeratio_jac=np.zeros(nlen)

        #runtimeratio_py=runtime_py/runtime_for
        for i in range(nlen):
            print(i)

            #Time Python implementation using Jacobi iteration
            t0=time.time()
            simulate_jacobi(nvalues[i])
            t1=time.time()
            runtimepython=t1-t0

            #Time Fortran implementation using Jacobi iteration
            t2=time.time()
            bm.simulate_jacobi(nvalues[i])
            t3=time.time()
            runtimefortran=t3-t2

            #Compute ratio of Jacobi method
            runtimeratio_jac[i]=runtimepython/runtimefortran

        #Plot of figure
        plt.hold(True)
        plt.plot(nvalues,runtimeratio_jac,'.',c='b',label='Python (Jacobi iteration method)')
        plt.xlabel('n')
        plt.ylabel('Ratio')
        plt.title('Manlin Chawla: performance(3) \n Ratio of runtimes (Python Runtime\Fortran Runtime) using Jacobi iteration, as N increases')
        #plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot of n agaist final deltac for Python implementations
    if fignum==4:
        #Set values of n (determines grid size)
        nlen=50
        nvalues=np.linspace(1,nlen,nlen).astype('int')

        #Preallocate for efficiency
        deltac_osi=np.zeros(nlen)
        deltac_jac=np.zeros(nlen)

        #Iterate over n and obtain deltac
        for i,n in enumerate(nvalues):
            print(n)

            #Call Python implementation using OSI method
            deltaC=simulate(n)[1]
            #Extarct final deltaC
            deltac_osi[i]=deltaC[-1]

            #Call Python implementation using Jacobi iteration method
            deltaC=simulate_jacobi(n)[1]
            deltac_jac[i]=deltaC[-1]

        #Plot of figure
        plt.hold(True)
        plt.plot(nvalues,deltac_osi,'.',c='r',label='Fortran (OSI method)')
        plt.plot(nvalues,deltac_jac,'.',c='b',label='Fortran (Jacobi method)')
        plt.xlabel('n')
        plt.ylabel('deltac')
        plt.legend()
        plt.title('Manlin Chawla: performance(4) \n Final value of deltac as n increases')
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot of delta C when using Jacobi iteration method and OSI iteration method
    if fignum==5:
        #Set values of n to be iterated over
        n=100
        bm.simulate(n)
        deltaC_osi=bm.deltac.copy()
        bm.simulate_jacobi(n)
        deltaC_jac=bm.deltac.copy()

        #Plot of figure
        plt.hold(True)
        plt.plot(deltaC_osi,'.',c='r',label='Python (OSI method)')
        plt.plot(deltaC_jac,'.',c='b',label='Python (Jac method)')
        plt.xlabel('Iteration')
        plt.ylabel('deltac')
        plt.legend()
        plt.title('Manlin Chawla: performance(5) \n deltac at each iteration, n=100')
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot 7: Comparison of runtime between Fortran OSI and Fortran Jacobi
    if fignum==6:
        #Set values of n (determines size of grid)
        nlen=50
        nvalues=np.linspace(1,nlen,nlen).astype('int')

        #Preallocate for efficiency
        runtime_osi=np.zeros(nlen)
        runtime_jac=np.zeros(nlen)
        #Repeat times and take minimum to get rid of fluctuations in Fortran runtimes
        timerepeat=np.zeros(25)

        for i,n in enumerate(nvalues):
            print(n)
            #Minimum runtime for Fortran implementation of Jacobi method
            for j in range(25):
                t0=time.time()
                bm.simulate_jacobi(n)
                t1=time.time()
                timerepeat[j]=t1-t0
            runtime_jac[i]=min(timerepeat)

            #Minimum runtime for Fortran implementation of OSI method
            for j in range(25):
                t2=time.time()
                bm.simulate(n)
                t3=time.time()
                timerepeat[j]=t3-t2
            runtime_osi[i]=min(timerepeat)

        #Plot of figure
        plt.hold(True)
        plt.plot(nvalues,runtime_jac,'.',c='r',label='Fortran (Jacobi method)')
        plt.plot(nvalues,runtime_osi,'.',c='b',label='Fortran (OSI method)')
        plt.xlabel('n')
        plt.ylabel('Runtime (s)')
        plt.title('Manlin Chawla: performance(6) \n Runtime of each method as n increases ')
        plt.legend()
        plt.hold(False)

        #Set display==false for plots to generate in if__name==main
        if display==True:
            plt.show()

    return None

def analyze(fignum,n,display=False):
    """Analyze influence of modified
    boundary condition on contamination dynamics
        Add input/output variables as needed.
    """
    """My approach to finding the best value of theta* was to create a new
    subroutine called simulate_antibac. This subroutine is a modified version of
    the simulate subroutine which uses the OSI method. The new subroutine takes
    in input parameters (bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n) and a new variable tstar,
    returns C and has a modified contamination boundary to include the new boundary
    conditions.

    To investigate the problem, I initially used the subroutine simulate_antibac
    to vary the values of theta* and see the effects on the contamination C.
    Analyze(1) is a plot of theta* against the average value of the contamination
    (when parameters are set to g=1, r0=1+pi/2,k=2,s0=2). The plot shows that the
    average contamination of C takes its lowest value at theta* =0 and theta*=pi.
    The plot shows an M shape as theta* is varied. After seeing the plot I added
    code to analyze(1) which outputs the value of theta* at the minima. The
    concentration of bacteria has a minima at pi/2 (or 1.5707963...).

    To investigate further I varied other parameters to check this was the case. Analyze(2),
    analyze(3), analyze(4) are plots of theta* against average value of contamination
    as g, r0 and theta0 are varied respectively. All three plots show the same
    trend where (even though values of C average are shifted) the lowest values
    are at theta*=0 and theta*=pi and there is a minima at theta*=pi/2.

    Analyze(8) is a plot of theta* against average value of contamination as k is varied.
    This shows that the average concentration of bacteria has an oscillation like trend
    of different frequencies as k is changed. K is the model parameter and appears
    in the boundary condition in the sin^2(k*theta) term. When k is an odd integer
    the average concentration of bacteria has maximums at theta*=pi/2, this can be explained
    by the property of sine.  When k is an even integer the average concentration of
    bacteria has a minimums at theta*=pi/2. Similar to analyze(2), analyze(3), analyze(4)
    for this plot each line has the lowest Cave at the end point theta*=0 and theta*=pi.
    Whether theta*=pi/2 is the best value to locally minimize the concentration of bacteria depends
    on k.

    Since these values keep on reoccuring, I used the contour plotting method
    from the simulate_jacobi function to visualize the final concentration level.
    Analyze(5), analyze(6), analyze(7) show these plots.

    For theta*=0 the plot shows that the bacteria is concentrated at a particular
    region and there is a smaller region of bacteria in the corner of the plot
    (at r=1, theta=0).The plot for theta*=pi is similar and again shows that the
    bacteria is concentrated at a particular region and there is a smaller region
    of bacteria in the corner of the plot (at close to r=1, theta=pi). The plot for
    theta*=pi/2 is a special case. It shows a concentrated region of bacteria
    but there are now two smaller regions of bacteria on the boundary r=1 close to theta=pi/2.
    All three plots show that the bacteria is concentrated at a region, and there is
    more contamination on the boundary at the value of theta*. However these
    areas can be isolated to contain the bacteria and stop contamination.

    Considering the modified boundary conditons at r=1, the exponential terms acts
    as a Gaussian model. This term will be largest when theta=theta* but at these
    values the sine terms becomes 0 as sin(k*pi)=sin(0)=0.

    So overall, considering all of the plots, the bacteria contamination is
    minimized at theta*=0 and theta*=pi, making these the best values of theta* 
    for the new model."""
    #Plot of the average value of C for different values of theta*
    if fignum==1:

        #Set input parameters
        bm_g=1
        bm_r0=1+np.pi/2
        bm_kbc=2
        bm_s0=2
        bm_t0=1.5

        #Number of points in the plot
        numpoints=100

        #Set up a list of values of theta* to use in subroutine simulate_antibac
        tstarvalues=np.linspace(0,np.pi,numpoints)

        #Preallocate for efficiency
        C_ave=np.zeros(numpoints)

        # Each iteration sets a new value of theta*
        for i,tstar in enumerate(tstarvalues):
            #Call subroutine with modified boundary conditions and inputs
            C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)
            #Calculate average of C vector
            C_ave[i]=C.mean()

        #Plot
        plt.hold(True)
        plt.plot(tstarvalues,C_ave,'.',c='r')
        plt.xlabel('theta star')
        plt.ylabel('Average value of C')
        plt.title('Manlin Chawla: analyze(1) \n Average of C against tstar')

        if display==True:
            plt.show()

        #Find value of theta* at the minimum turning point (minima)
        for i in np.linspace(1,len(C_ave)-2,len(C_ave)-2).astype(int):
            if (C_ave[i]<C_ave[i+1]) & (C_ave[i-1]>C_ave[i]):
                print(i)
                print(tstarvalues[i])

    #Plot for values of tstar against C_ave as g is varied
    if fignum==2:

        #Set input parameters
        bm_r0=1+np.pi/2
        bm_kbc=2
        bm_s0=2
        bm_t0=1.5

        #Set number of points in the plot
        numpoints=100

        #Set up list of theta* and g values
        tstarvalues=np.linspace(0,np.pi,numpoints)
        gvalues=np.linspace(1,10,4)

        #Preallocate for efficiency
        C_ave=np.zeros(numpoints)

        #Iterate over gvalues
        for g in gvalues:
            #Set current value of g to use as input in the subroutine
            bm_g=g

            plt.hold(True)

            #Iterate over theta* values
            for i,tstar in enumerate(tstarvalues):
                #Call subroutine
                C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)
                #get average of the vector C
                C_ave[i]=C.mean()
            #Plot a line for each value of g
            plt.plot(tstarvalues,C_ave,'.',markersize='4',label='g='+str(g))

        plt.xlabel('theta star')
        plt.ylabel('Average value of C')
        plt.title('Manlin Chawla: analyze(2) \n Average of C against tstar for different values of g')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot for values of tstar against C_ave as r is varied
    if fignum==3:

        #Setup input parameters
        bm_g=1
        bm_kbc=2
        bm_s0=2
        bm_t0=1.5

        #Number of points in the plot
        numpoints=100

        #Set up theta* and r0 values to iterate over
        tstarvalues=np.linspace(0,np.pi,numpoints)
        r0values=np.linspace(1,1+np.pi,4)

        #Preallocate for efficiency
        C_ave=np.zeros(numpoints)

        #Iterate of r0 values
        for r0 in r0values:
            #Set current vlaue of r0 as input parameter for subroutine
            bm_r0=r0

            plt.hold(True)

            #Iterate of theta* values
            for i,tstar in enumerate(tstarvalues):
                #Call subroutine
                C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)
                #get average of C vector
                C_ave[i]=C.mean()
            #Plot a line for every line of r0
            plt.plot(tstarvalues,C_ave,'.',markersize='4',label='r0='+str(round(r0,4)))

        plt.xlabel('theta star')
        plt.ylabel('Average value of C')
        plt.title('Manlin Chawla: analyze(3) \n Average of C against tstar for different values of r0')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot for values of tstar against C_ave as theta0 is varied
    if fignum==4:

        #Setup input parameters
        bm_g=1
        bm_kbc=2
        bm_s0=2
        bm_r0=1+np.pi/2

        #Number of points in the plot
        numpoints=100

        #Set up theta* and r0 values to iterate over
        tstarvalues=np.linspace(0,np.pi,numpoints)
        t0values=np.linspace(0,np.pi,5)

        #Preallocate for efficiency
        C_ave=np.zeros(numpoints)

        #Iterate of r0 values
        for t0 in t0values:
            #Set current vlaue of r0 as input parameter for subroutine
            bm_t0=t0

            plt.hold(True)

            #Iterate of theta* values
            for i,tstar in enumerate(tstarvalues):
                #Call subroutine
                C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)
                #get average of C vector
                C_ave[i]=C.mean()
            #Plot a line for every line of r0
            plt.plot(tstarvalues,C_ave,'.',markersize='4',label='t0='+str(round(t0,4)))

        plt.xlabel('theta star')
        plt.ylabel('Average value of C')
        plt.title('Manlin Chawla: analyze(4) \n Average of C against tstar for different values of theta0')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot for values of tstar against C_ave as k is varied
    if fignum==8:

        #Setup input parameters
        bm_g=1
        #bm_kbc=2
        bm_s0=2
        bm_t0=1.5
        bm_r0=1+np.pi/2

        #Number of points in the plot
        numpoints=100

        #Set up theta* and k values to iterate over
        tstarvalues=np.linspace(0,np.pi,numpoints)
        kvalues=np.linspace(1,6,6)

        #Preallocate for efficiency
        C_ave=np.zeros(numpoints)

        #Iterate of r0 values
        for k in kvalues:
            #Set current vlaue of r0 as input parameter for subroutine
            bm_kbc=k

            plt.hold(True)

            #Iterate of theta* values
            for i,tstar in enumerate(tstarvalues):
                #Call subroutine
                C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)
                #get average of C vector
                C_ave[i]=C.mean()
            #Plot a line for every line of r0
            plt.plot(tstarvalues,C_ave,'.',markersize='4',label='k='+str(round(k,4)))

        plt.xlabel('theta star')
        plt.ylabel('Average value of C')
        plt.title('Manlin Chawla: analyze(8) \n Average of C against tstar for different values of k')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot of contamination at tstar=0
    if fignum==5:

        #Set input parameters
        bm_g=1
        bm_r0=1+np.pi/2
        bm_kbc=2
        bm_s0=2
        bm_t0=1.5

        #Set numerical parameters
        r = np.linspace(1,1+np.pi,n+2)
        t = np.linspace(0,np.pi,n+2) #theta
        tg,rg = np.meshgrid(t,r) # r-theta grid
        #tstar=np.pi/2
        #tstar=np.pi
        tstar=0

        C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)

        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Manlin Chawla: analyze(5) \n Final concentration field tstar=0')

        if display==True:
            plt.show()

    #Plot of contamination at tstar=pi/2
    if fignum==6:

        #Set input parameters
        bm_g=1
        bm_r0=1+np.pi/2
        bm_kbc=2
        bm_s0=2
        bm_t0=1.5

        #Set numerical parameters
        r = np.linspace(1,1+np.pi,n+2)
        t = np.linspace(0,np.pi,n+2) #theta
        tg,rg = np.meshgrid(t,r) # r-theta grid
        tstar=np.pi/2
        #tstar=np.pi
        #tstar=0

        C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)

        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Manlin Chawla: analyze(6) \n Final concentration field tstar=pi/2')

        if display==True:
            plt.show()

    #Plot of contamination at tstar=pi
    if fignum==7:

        #Set input parameters
        bm_g=1
        bm_r0=1+np.pi/2
        bm_kbc=2
        bm_s0=2
        bm_t0=1.5

        #Set numerical parameters
        r = np.linspace(1,1+np.pi,n+2)
        t = np.linspace(0,np.pi,n+2) #theta
        tg,rg = np.meshgrid(t,r) # r-theta grid
        #tstar=np.pi/2
        tstar=np.pi
        #tstar=0.5

        C=bm.simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar)

        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Manlin Chawla: analyze(7) \n Final concentration field tstar=pi')

        if display==True:
            plt.show()

    return None


if __name__=='__main__':
    #Add code below to call performance and analyze
    #and generate figures you are submitting in
    #your repo.
    output_p = performance(1)
    plt.savefig('p31.png', bbox_inches="tight")
    plt.clf()
    print('performance(1) plot saved')

    output_p = performance(2)
    plt.savefig('p32.png', bbox_inches="tight")
    plt.clf()
    print('performance(2) plot saved')

    output_p = performance(3)
    plt.savefig('p33.png', bbox_inches="tight")
    plt.clf()
    print('performance(3) plot saved')

    output_p = performance(4)
    plt.savefig('p34.png', bbox_inches="tight")
    plt.clf()
    print('performance(4) plot saved')

    output_p = performance(5)
    plt.savefig('p35.png', bbox_inches="tight")
    plt.clf()
    print('performance(5) plot saved')

    output_p = performance(6)
    plt.savefig('p36.png', bbox_inches="tight")
    plt.clf()
    print('performance(6) plot saved')

    output_a = analyze(1,100)
    plt.savefig('p41.png', bbox_inches="tight")
    plt.clf()
    print('analyze(1) plot saved')

    output_a = analyze(2,100)
    plt.savefig('p42.png', bbox_inches="tight")
    plt.clf()
    print('analyze(2) plot saved')

    output_a = analyze(3,100)
    plt.savefig('p43.png', bbox_inches="tight")
    plt.clf()
    print('analyze(3) plot saved')

    output_a = analyze(4,100)
    plt.savefig('p44.png', bbox_inches="tight")
    plt.clf()
    print('analyze(4) plot saved')

    output_b = analyze(5,100)
    plt.savefig('p45.png', bbox_inches="tight")
    plt.clf()
    print('analyze(5) plot saved')

    output_c = analyze(6,100)
    plt.savefig('p46.png', bbox_inches="tight")
    plt.clf()
    print('analyze(6) plot saved')

    output_d = analyze(7,100)
    plt.savefig('p47.png', bbox_inches="tight")
    plt.clf()
    print('analyze(7) plot saved')

    output_d = analyze(8,100)
    plt.savefig('p48.png', bbox_inches="tight")
    plt.clf()
    print('analyze(8) plot saved')
