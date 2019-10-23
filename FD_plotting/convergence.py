import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc

# Plot up errors at final time by measuring the differece between the solution found with AIR and the exact solution.

# Just errors I copied and pasted from terminal
e2 = np.array([
        [4.6970e-02, 4.4682e-02, 2.8023e-02, 1.4964e-02, 7.8854e-03, 4.0085e-03, 2.0209e-03], 
        [2.2664e-01, 9.0404e-02, 2.4382e-02, 6.1499e-03, 1.5442e-03, 3.8523e-04], 
        [1.0561e-01, 1.8453e-02,  2.4385e-03, 3.0417e-04, 3.7475e-05], 
        [2.4745e-02, 1.6690e-03, 1.0577e-04, 5.8700e-06, 8.6072e-07, 8.9262e-07]
        ]) 
        
colours = ["b", "r", "g", "c"]        

nx = 2**np.arange(5,12)

for i in range(0,4):
    plt.loglog(nx[:len(e2[i])], e2[i], label = "RK{}+U{}".format(i+1, i+1), marker = "o", color = colours[i])
    plt.loglog(nx[:len(e2[i])], 0.5*e2[i][-1]*(nx[:len(e2[i])]/float(nx[len(e2[i])-1]))**(-i-1), linestyle = '--', color = colours[i])

fs = 18
plt.legend(fontsize = fs-4)
plt.title("$\\sqrt{{\\Delta x}} \\Vert u_{{\\rm{{exact}}}}(x,1) - u_{{\\rm{{num}}}}(x,1) \\Vert_2$", fontsize = fs)
plt.xlabel("$n_x$", fontsize = fs)
plt.savefig('advection1D_convergence.pdf', bbox_inches='tight')
#plt.show()    


