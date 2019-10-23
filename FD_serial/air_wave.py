import pdb
import time
import math
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, identity
from scipy.io import mmread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from pyamg.classical.air import AIR_solver
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.util.utils import scale_block_inverse, get_block_diag
from upwind_wave import get_upwind_wave, get_upwind_wave_2d
from dolfin_disc import get_supg_advection_diffusion, get_dg_adv_diff_react, \
	get_supg_adv_diff_react, get_poisson, get_DG_transport


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20 			# Max levels in hierarchy
max_coarse 		   = 20 		# Max points allowed on coarse grid
tol 			   = 1e-12		# Residual convergence tolerance
is_pdef 		   = False		# Assume matrix positive definite (only for aSA)
keep_levels 	   = False		# Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel =  None
keep = False
cycle = 'V'


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# nx = 50
nx = 4
hx = 1.0/nx
nt = 2
# nt = zz*50
dt = 1.0/nt
c = 0.495*hx/dt/1.5
c = 1.0
dim = 1

if dim == 1:
	A, b = get_upwind_wave(nx=nx,nt=nt,dt=dt,c=c)
else:
	A, b = get_upwind_wave_2d(nx=nx, ny=nx, nt=nt, dt=dt, cx=c, cy=c/1.0)

pdb.set_trace()

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# b = np.zeros((A.shape[0],))
# x0 = np.random.rand(A.shape[0],)
x0 = np.ones((A.shape[0],))
init_nnz = A.nnz

inds = np.where(b > 0)[0]
# b[(400*2*nx*nx):(400*2*nx*nx+20)] = 1


bsr = False
if not bsr:
	try:
		bsize = A.blocksize[0]
	except:
		bsize = 1
	# A, Dinv = scale_block_inverse(A, blocksize=bsize)
	# b = Dinv * b	
	A = A.tocsr()
	A.eliminate_zeros()
	bsize = None
else:
	bsize = A.blocksize[0]

if bsr:
	strength = ('classical', {'theta': 0.1, 'norm': 'abs', 'block': 'block'} )
else:
	strength = ('classical', {'theta': 0.1, 'norm': 'min', 'block': None} )
splitting = ('RS', {'second_pass': True})
# splitting = 'PMIS'
# interp = ('distance_two', {'theta': 0.2})
# interp = ('standard', {'theta': 0.1, 'modified': True})
interp = 'one_point'
coarse_grid_P = None

if bsr:
	# restrict = ('air', {'theta': 0.2, 'degree': 1, 'use_gmres': False, 'precondition': True, 'maxiter': 2})
	restrict = ('neumann', {'theta': 0.2, 'degree': 1})
else:
	restrict = ('air', {'theta': 0.1, 'degree': 1, 'use_gmres': False, 'precondition': True, 'maxiter': 5})
	# restrict = ('neumann', {'theta': 0.05, 'degree': 1})
	# coarse_grid_R = 'inject'
	# coarse_grid_R = ('air', {'theta': 0.1, 'degree': 2, 'use_gmres': False, 'precondition': True, 'maxiter': 5})

# restrict = 'trivial'
# restrict = 'inject'
# interp = ('air', {'theta': 0.1, 'degree': 1, 'use_gmres': False, 'precondition': True, 'maxiter': 5})
# coarse_grid_P = interp
# restrict = None
# interp = 'restrict'


coarse_grid_R = None
# filter_operator = (True, 1e-4)
filter_operator = None

# Relaxation
# ----------
block = bsize
omega = 3.0/3.0		# withrho omega = 5/3 appears best, equivalent to unweighted Jacobi w/o rho
withrho = False
C_iter1 = 0
F_iter1 = 0
C_iter2 = 0
F_iter2 = 2
if bsr:
	relaxation1 = ('CF_block_jacobi', {'omega': omega, 'iterations': 1, 'blocksize': block,
						'withrho': withrho, 'F_iterations': F_iter1, 'C_iterations': C_iter1} )
	relaxation2 = ('FC_block_jacobi', {'omega': omega, 'iterations': 1, 'blocksize': block,
						'withrho': withrho, 'F_iterations': F_iter2, 'C_iterations': C_iter2} )
else:
	relaxation1 = ('CF_jacobi', {'omega': omega, 'iterations': 1, 'withrho': withrho,
				     'F_iterations': F_iter1, 'C_iterations': C_iter1} )
	relaxation2 = ('FC_jacobi', {'omega': omega, 'iterations': 1, 'withrho': withrho,
				     'F_iterations': F_iter2, 'C_iterations': C_iter2} )
	# relaxation1 = ('gauss_seidel', {'iterations': 1, 'sweep': 'backward'})
	# relaxation2 = ('jacobi', {'omega': 2.0/3.0, 'iterations': 1, 'withrho': withrho})

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Airhead solver
# -------------------
start = time.clock()
ml = AIR_solver(A=A, strength=strength, CF=splitting, interp=interp, restrict=restrict,
               	    presmoother=relaxation1, postsmoother=relaxation2,
               	    max_levels=max_levels, max_coarse=max_coarse, keep=keep,
               	    coarse_grid_P=coarse_grid_P, coarse_grid_R=coarse_grid_R,
               	    filter_operator=filter_operator)
end = time.clock()
setup_time = end-start

print "AMGir: ",A.shape[0]," DOFs, ",A.nnz," nnzs"
print "\tSetup time 	 	 = ",setup_time

residuals = []
start = time.clock()
sol = ml.solve(b, x0, tol, residuals=residuals, accel=accel, cycle=cycle, maxiter=250)
end = time.clock()
solve_time = end-start

OC = ml.operator_complexity()
CC = ml.cycle_complexity(cycle=cycle)
SC = ml.setup_complexity()

# Average convergence factor
if residuals[-1] == 0:
	residuals[-1] = 1e-16
elif residuals[-1] > 1e2:
	residuals[-1] = residuals[0]
	residuals[-2] = residuals[0]

CF = (residuals[-1]/residuals[0])**(1.0/(len(residuals)-1))
last_CF = residuals[-1]/residuals[-2]
scale = float(A.nnz) / init_nnz

if CF == 1:
	WPD = -1
else:
	WPD = -CC / np.log10(CF)

# All Convergence factors
conv_factors = np.zeros((len(residuals)-1,1))
for i in range(0,len(residuals)-1):
	conv_factors[i] = residuals[i+1]/residuals[i]

print "\tSolve time 	 	 = ",solve_time
print "\tSetup complexity 	 = ",SC * scale
print "\tOperator complexity 	 = ",OC * scale
print "\tCycle complexity 	 = ",CC * scale
print "\tWork per digit 		 = ",WPD * scale
print "\tAverage CF 	 	 = ",CF
print "\tFinal CF 	 	 = ",last_CF
print "\tCFL condition 		 = ",c*dt/hx
print "\tIterations		= ",len(conv_factors)
print "\tFinal res		= ",residuals[-1]
# print nt,",",nx,",",A.shape[0],",",A.nnz,",",len(conv_factors),",",OC,",",CC,",",c


# Plot wave equation
plotsol = True
plotres = True
fontsize = 20
params = {'backend': 'ps',
          'axes.labelsize': fontsize,
          'font.size': fontsize,
          'axes.titlesize' : fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}
plt.rcParams.update(params)
if plotsol:
	for i in range(0,nt,nt/10):
		plot_tind = i
		plot_ind0 = plot_tind*nx*nx*2
		plot_ind1 = (plot_tind+1)*nx*nx*2
		test_wave = sol[plot_ind0:plot_ind1:2]
		test_wave = test_wave.reshape((nx,nx))
		X = np.arange(0, 1.0, hx)
		Y = np.arange(0, 1.0, hx)
		X, Y = np.meshgrid(X, Y)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(X, Y, test_wave, cmap=cm.coolwarm, 
		                       antialiased=False, alpha=1.0, rstride=1, cstride=1, linewidth=1)
		ax.set_zlim(0, 1.2)
		plt.tight_layout()
		plt.savefig('waveSol_t' + str(i) + '.pdf', transparent=True)

if plotres:
	fig1 = plt.figure()
	# Create left axis plots
	axL = fig1.add_subplot(111, ylim=(1e-12, 2))
	axL.yaxis.tick_left()
	line3, = plt.plot(np.arange(1,len(residuals))-0.5,conv_factors, 'b-o', markersize=7, label='$\mathbf{r}_i$')
	plt.ylabel("Convergence factor")
	plt.xlabel("Iteration number")
	axL.grid()
	# axL.legend(handles=[line1,line2],ncol=1,loc='lower left')

	axR = fig1.add_subplot(111, sharex=axL, ylim=(1e-16,10**(np.ceil(np.log10(np.max(residuals))))))
	axR.yaxis.tick_right()
	axR.yaxis.set_label_position("right")
	line1, = plt.semilogy(np.arange(0,len(residuals)),residuals, 'k-o', markersize=7, label='$\mathbf{r}_i$')
	plt.ylabel("Residual")
	# plt.legend(handles=[line3,line4],ncol=1,loc='lower right')
	plt.savefig('wave_res_CF.pdf', transparent=True)


pdb.set_trace()

