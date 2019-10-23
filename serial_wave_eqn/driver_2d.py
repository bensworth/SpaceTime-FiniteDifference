import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular
import scipy.sparse.linalg as spla

#import upwind_wave as uw
import banks_upwind_wave as buw


def heaviside(x):
	if x < 0:
		return 0
	else:
		return 1


# Initial condition
# def u0(x, y):
# 	return	heaviside(x + 0.25)*heaviside(y + 0.25) - \
# 			heaviside(x + 0.25)*heaviside(y - 0.25) - \
# 			heaviside(x - 0.25)*heaviside(y + 0.25) + \
# 			heaviside(x - 0.25)*heaviside(y - 0.25)

def u0(x, y):
	return np.sin(np.pi*y) #np.sin(np.pi*x) #*np.sin(np.pi*y)
	#return heaviside(x + 0.25) - heaviside(x - 0.25)


# Problem parameters
# Tmax = 2.5
# nx = 24
# ny = 24
# cx = 1
# cy = 1
# CFL_saftety = 0.9

Tmax = 0.1   
nx = 6
ny = 6
cx = 1
cy = 1
CFL_saftety = 0.9


# Spatial mesh
x = np.linspace(-1, 1, nx+1)
x = x[:-1] # nx points in [-1, 1)
hx = x[1] - x[0]
y = np.linspace(-1, 1, ny+1)
y = y[:-1] # ny points in [-1, 1)
hy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Evalute IC at (xi,yj)-th grid point
def u0_wrapper(xi, yj):
	return u0(x[xi], y[yj])


### --------------- 1st-order scheme --------------- ###
# scheme = "UW1"
# dt = CFL_saftety * buw.get_CFL_limit_2d(scheme, hx, cx, hy, cy)
# t = dt*np.arange(0, np.ceil(Tmax/dt)+1)
# nt = t.shape[0]
# print("{}: nt = {}, dt = {:.2e}, nx = {}, hx = {:.2e}, cx = {}, ny = {}, hy = {:.2e}, cy = {}".format(scheme, nt, dt, nx, hx, cx, ny, hy, cy))
# print("{}: assembling...".format(scheme))
# A, b = buw.get_UW1_2D(nt, dt, nx, hx, cx, ny, hy, cy, u0_wrapper)
# A = A.tocsr()
# print("{}: solving...\n".format(scheme))
# z = spsolve_triangular(A, b, lower = True)
# u = z[0::2].copy()
# #v = z[1::2].copy()
# 
# print(u.shape, u[nx*ny*(nt - 1):].shape, nx, ny, nt, X.shape, Y.shape)
# 
# fig = plt.figure(1)
# ax = fig.gca(projection='3d') 
# cmap = plt.cm.get_cmap("coolwarm")
# surf = ax.plot_surface(X, Y, u[nx*ny*(nt - 1):].reshape(ny, nx), cmap = cmap)
# plt.title("UW1-2D: u(x,y, t = {:.2f})".format(t[-1]), fontsize = 15)
# plt.xlabel("x", fontsize = 15)
# plt.ylabel("y", fontsize = 15)



### --------------- 2nd-order scheme --------------- ###
scheme = "UW2"
dt = CFL_saftety * buw.get_CFL_limit_2d(scheme, hx, cx, hy, cy)
t = dt*np.arange(0, np.ceil(Tmax/dt)+1)
nt = t.shape[0]
print("{}: nt = {}, dt = {:.2e}, nx = {}, hx = {:.2e}, cx = {}, ny = {}, hy = {:.2e}, cy = {}".format(scheme, nt, dt, nx, hx, cx, ny, hy, cy))
print("{}: assembling...".format(scheme))
A, b = buw.get_UW2_2D(nt, dt, nx, hx, cx, ny, hy, cy, u0_wrapper)
A = A.tocsr()
print("{}: solving...".format(scheme))
z = spsolve_triangular(A, b, lower = True)
u = z[0::2].copy()
#v = z[1::2].copy()

plt.spy(A)

# s = 2 # How many times to sample solution
# for i in range(0, np.int(np.floor(nt/s))):
# 	fig = plt.figure(i+2)
# 	cmap = plt.cm.get_cmap("coolwarm")
# 
# 	U = u[nx*ny*s*i:nx*ny*(s*i+1)].reshape(ny, nx)
# 	ax = fig.gca(projection='3d') 
# 	surf = ax.plot_surface(X, Y, U, cmap = cmap)
# 
# 	# U = u[nx*ny*s*i:nx*ny*(s*i+1)].reshape(ny, nx)
# 	# levels = np.linspace(np.amin(U, axis = (0,1)), np.amax(U, axis = (0,1)), 100)
# 	# pl = plt.contourf(X, Y, U, levels = levels, cmap = cmap)
# 	#plt.colorbar(ticks=np.linspace(np.amin(U), np.amax(U), 7), format='%0.1f')
# 	# 
# 	plt.title("UW2-2D: u(x,y, t = {:.2f})".format(t[s*i]), fontsize = 15)
# 	plt.xlabel("x", fontsize = 15)
# 	plt.ylabel("y", fontsize = 15)



# fig = plt.figure(2)
# ax = fig.gca(projection='3d') 
# cmap = plt.cm.get_cmap("coolwarm")
# surf = ax.plot_surface(X, Y, u[nx*ny*(nt - 1):].reshape(ny, nx), cmap = cmap)
# plt.title("UW2-2D: u(x,y, t = {:.2f})".format(t[-1]), fontsize = 15)
# plt.xlabel("x", fontsize = 15)
# plt.ylabel("y", fontsize = 15)



plt.show()