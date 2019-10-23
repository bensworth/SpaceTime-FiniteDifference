import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pdb
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse.linalg import spsolve_triangular

#import upwind_wave as uw
import banks_upwind_wave as buw

def heaviside(x):
	if x < 0:
		return 0
	else:
		return 1
	
def u0(x):
	return heaviside(x + 0.25) - heaviside(x - 0.25)

def u0(x):
	return np.sin(np.pi*x)	

# Eact solution of wave eqn. Eq (15) assuming 0 initial velocity. But this is 
# on an infinite interval? I want the solution on a periodic interval...
# So I don't really understand how this is right some of the time...
def u_dalambert(t, x, c):
	return 0.5*(u0(x - c*t) + u0(x + c*t))
	
Tmax = 0.6
nx = 16
cx = 0.5
CFL_safety = 0.5


# Spatial mesh
x = np.linspace(-1, 1, nx+1)
x = x[:-1] # nx points in [-1, 1)
hx = x[1] - x[0]

# Evalute IC at xi-th grid point
def u0_wrapper(xi):
	return u0(x[xi])


### --------------- 1st-order scheme (1a) --------------- ###
scheme = "UW1a"
dt = CFL_safety * buw.get_CFL_limit_1d(scheme, hx, cx)
dt = 0.25
t = dt*np.arange(0, np.ceil(Tmax/dt)+1)
nt = t.shape[0]
print("{}: nt = {}, dt = {:.2e}, nx = {}, hx = {:.2e}, cx = {}\n".format(scheme, nt, dt, nx, hx, cx))
print("{}: assembling...\n".format(scheme))
A, b = buw.get_UW1a_1D(nt, dt, nx, hx, cx, u0_wrapper)
A = A.tocsr()
print("{}: solving...\n".format(scheme))
# z = spsolve_triangular(A, b, lower = True)
# u = z[0::2].copy()
#v = z[1::2].copy()

# plt.plot(x, u[nx*(nt-1)::], label = scheme)

matpath = "..//FD_class/Apy_UW1a.npz"
save_npz(matpath, A)

### --------------- 1st-order scheme --------------- ###
scheme = "UW1"
dt = CFL_safety * buw.get_CFL_limit_1d(scheme, hx, cx)
dt = 0.25
t = dt*np.arange(0, np.ceil(Tmax/dt)+1)
nt = t.shape[0]
print("lambda = {:.5f}".format(cx*dt/hx))
print("{}: nt = {}, dt = {:.2e}, nx = {}, hx = {:.2e}, cx = {}\n".format(scheme, nt, dt, nx, hx, cx))
print("{}: assembling...\n".format(scheme))
A, b = buw.get_UW1_1D(nt, dt, nx, hx, cx, u0_wrapper)
A = A.tocsr()
print("{}: solving...\n".format(scheme))
# z = spsolve_triangular(A, b, lower = True)
# u = z[0::2].copy()
#v = z[1::2].copy()

# plt.plot(x, u[nx*(nt-1)::], label = scheme)

matpath = "..//FD_class/Apy_UW1.npz"
save_npz(matpath, A)
# 
# ### --------------- 2nd-order scheme --------------- ###
scheme = "UW2"
dt = CFL_safety * buw.get_CFL_limit_1d(scheme, hx, cx)
dt = 0.25
t = dt*np.arange(0, np.ceil(Tmax/dt)+1)
nt = t.shape[0]
print("{}: nt = {}, dt = {:.2e}, nx = {}, hx = {:.2e}, cx = {}\n".format(scheme, nt, dt, nx, hx, cx))
print("{}: assembling...\n".format(scheme))
A, b = buw.get_UW2_1D(nt, dt, nx, hx, cx, u0_wrapper)
A = A.tocsr()
print("{}: solving...\n".format(scheme))
z = spsolve_triangular(A, b, lower = True)
u = z[0::2].copy()
#v = z[1::2].copy()

plt.plot(x, u[nx*(nt-1)::], label = scheme)
matpath = "..//FD_class/Apy_UW2.npz"
save_npz(matpath, A)

### --------------- 4th-order scheme --------------- ###
scheme = "UW4"
dt = CFL_safety * buw.get_CFL_limit_1d(scheme, hx, cx)
dt = 0.25
t = dt*np.arange(0, np.ceil(Tmax/dt)+1)
nt = t.shape[0]
print("{}: nt = {}, dt = {:.2e}, nx = {}, hx = {:.2e}, cx = {}\n".format(scheme, nt, dt, nx, hx, cx))
print("{}: assembling...\n".format(scheme))
A, b = buw.get_UW4_1D(nt, dt, nx, hx, cx, u0_wrapper)
A = A.tocsr()
print("{}: solving...\n".format(scheme))
z = spsolve_triangular(A, b, lower = True)
u = z[0::2].copy()
#v = z[1::2].copy()

plt.plot(x, u[nx*(nt-1)::], label = scheme)
matpath = "..//FD_class/Apy_UW4.npz"
save_npz(matpath, A)
# 
# 
# # Get exact solution
# uexact = np.zeros_like(x)
# for i in range(0,nx):
# 	uexact[i] = u_dalambert(t[-1], x[i], cx)
# plt.plot(x, uexact, 'ko', label = "exact")
# 
# plt.legend()
# plt.title("u(x, t = {:.2f})".format(t[-1]), fontsize = 15)
# plt.xlabel("x", fontsize = 15)
# plt.show()


# fig = plt.figure(2)
# ax = fig.gca(projection='3d') 
# cmap = plt.cm.get_cmap("coolwarm")
# X, T = np.meshgrid(x, t)
# surf = ax.plot_surface(X, T, u.reshape((nt, nx)), cmap = cmap)
# plt.title("u4(x,t)", fontsize = 15)
# plt.xlabel("x", fontsize = 15)
# plt.ylabel("t", fontsize = 15)
# plt.show()




