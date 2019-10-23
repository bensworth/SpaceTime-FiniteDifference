import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np
import pdb




# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# One-dimensional schemes


# For given problem parameters, compute maximum allowable time step size, dt.
def get_CFL_limit_1d(scheme, hx, cx):
	if scheme == "UW1a":
		return (1 + np.sqrt(5))/4*hx/cx
	elif scheme == "UW1":
		return hx/cx
	elif scheme == "UW2":
		return 0.5*(np.sqrt(5) - 1)*hx/cx
	elif scheme == "UW4":
		return 1.09*hx/cx


# 1st-order scheme
def get_UW1a_1D(nt, dt, nx, hx, c, u0):
	K = c * dt / hx
	
	# Stencil coefficients
	cuu = np.array([1/8*(4*K*K + 1), 1/4*(3 - 4*K*K), 1/8*(4*K*K + 1)]) # u_i^(n+1) to u^n 
	cuv = np.array([dt*K/4, dt/2*(2 - K), dt*K/4]) # u_i^(n+1) to v^n 
	cvu = np.array([K*K/dt, -2*K*K/dt, K*K/dt]) # v_i^(n+1) to v^n 
	cvv = np.array([K/2, 1 - K, K/2]) # v_i^(n+1) to v^n 
	
	print()
	print(np.sum(cuu))
	print(np.sum(cuv))
	print(np.sum(cvu))
	print(np.sum(cvv))
	
	return get_1d_system(nt, nx, [cuu, cuv, cvu, cvv], u0)
	
	
# 1st-order scheme
def get_UW1_1D(nt, dt, nx, hx, c, u0):
	K = c * dt / hx
	
	# Stencil coefficients
	cuu = np.array([1.0/2.0*K*K, (1.0 - K*K), 1.0/2.0*K*K]) # u_i^(n+1) to u^n 
	cuv = np.array([dt*K/4.0, dt/2.0*(2.0 - K), dt*K/4.0]) # u_i^(n+1) to v^n 
	cvu = np.array([K*K/dt, -2.0*K*K/dt, K*K/dt]) # v_i^(n+1) to v^n 
	cvv = np.array([K/2.0, 1.0 - K, K/2.0]) # v_i^(n+1) to v^n 
	
	return get_1d_system(nt, nx, [cuu, cuv, cvu, cvv], u0)
	
	
# 2nd-order scheme
def get_UW2_1D(nt, dt, nx, hx, c, u0):
	K = c * dt / hx

	# Stencil coefficients
	gu = np.array([ K**3.0/(4.0*dt), 
					K**2.0/dt*(1.0 - K), 
				   -K**2.0/(2.0*dt)*(4.0 - 3.0*K), 
					K**2.0/dt*(1.0 - K), 
					K**3.0/(4.0*dt)])
	gv = np.array([-K/8.0, 
					K/2.0*(1.0 + K), 
				   -K/4.0*(3.0 + 4.0*K), 
					K/2.0*(1.0 + K), 
				   -K/8.0])

	cuu = dt/2.0*gu.copy()
	cuu[2] += 1
	cuv = dt/2.0*gv.copy()
	cuv[2] += dt

	cvu = gu.copy()
	cvv = gv.copy()
	cvv[2] += 1
	
	return get_1d_system(nt, nx, [cuu, cuv, cvu, cvv], u0)
	
	
# 4th-order scheme
def get_UW4_1D(nt, dt, nx, hx, c, u0):
	K = c * dt / hx

	# Stencil coefficients
	cuu = np.array([-K**3/432.0*( 9.0           - 2.0*K**2), 
	 				-K**2/72.0*(  3.0  - 9.0*K  - 3.0*K**2  + 2.0*K**3), 
					 K**2/144.0*(96.0 - 45.0*K - 24.0*K**2 + 10.0*K**3), 
					 1.0/108.0*(  108.0         - 135.0*K**2 + 45.0*K**3 + 27.0*K**4 - 10.0*K**5), 
					 K**2/144.0*(96.0 - 45.0*K - 24.0*K**2 + 10.0*K**3),
					-K**2/72.0*(  3.0  - 9.0*K  - 3.0*K**2  + 2.0*K**3), 
					-K**3/432.0*( 9.0           - 2.0*K**2),
					])
	cuv = np.array([ dt*K/576.0*(   5.0            -  4.0*K**2), 
					-dt*K/864.0*(  45.0  + 12.0*K  - 36.0*K**2  - 8.0*K**3), 
					 dt*K/1728.0*(225.0 + 384.0*K - 180.0*K**2 - 64.0*K**3), 
					 dt/144.0*(   144.0  - 25.0*K  - 60.0*K**2 + 20.0*K**3 + 8.0*K**4), 
					 dt*K/1728.0*(225.0 + 384.0*K - 180.0*K**2 - 64.0*K**3),
					-dt*K/864.0*(  45.0  + 12.0*K  - 36.0*K**2  - 8.0*K**3), 
					 dt*K/576.0*(   5.0             - 4.0*K**2)
					])
	cvu = np.array([-K**3/(48.0*dt)*( 3.0               - K**2), 
	 				-K**2/(24.0*dt)*( 2.0  - 9.0*K  - 4.0*K**2 +  3.0*K**3), 
					 K**2/(48.0*dt)*(64.0 - 45.0*K - 32.0*K**2 + 15.0*K**3), 
					-K**2/(12.0*dt)*(30.0 - 15.0*K - 12.0*K**2 +  5.0*K**3), 
					 K**2/(48.0*dt)*(64.0 - 45.0*K - 32.0*K**2 + 15.0*K**3), 
					-K**2/(24.0*dt)*( 2.0  - 9.0*K  - 4.0*K**2 +  3.0*K**3), 
					-K**3/(48.0*dt)*( 3.0               - K**2)])
	cvv = np.array([ K/288.0*(5.0           - 8.0*K**2), 
					-K/48.0*( 5.0  + 2.0*K  - 8.0*K**2  - 2.0*K**3), 
					 K/96.0*(25.0 + 64.0*K - 40.0*K**2 - 16.0*K**3), 
					 1.0/72.0*(72.0 - 25.0*K - 90.0*K**2 + 40.0*K**3 + 18.0*K**4), 
					 K/96.0*(25.0 + 64.0*K - 40.0*K**2 - 16.0*K**3), 
					-K/48.0*( 5.0  + 2.0*K  - 8.0*K**2  - 2.0*K**3), 
					 K/288.0*(5.0           - 8.0*K**2)])

	print(-cuu)
	print()
	print(-cuv)
	
	print()
	print()
	print(-cvv)
	print()
	print(-cvu)

	return get_1d_system(nt, nx, [cuu, cuv, cvu, cvv], u0)
	
		
		
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Space-time system for arbitrary-order scheme, one spatial dimension.
# Assume global ordering:
#	{ (x0,t0), (x1,t0), ..., (xn,t0), (x1,t1),...,
#	  (xn,t1), ..., (x0,tn),..., (xn,tn) }
#
# Assumes that spatial stencils are centred.
def get_1d_system(nt, nx, coefficients, u0):
	
	# Return global index for variable u at node (ti,xi) 
	def get_u(ti, xi):
		return 2*ti*nx + 2*((xi + nx) % nx)

	# Return global index for variable v at node (ti,xi) 
	def get_v(ti, xi):
		return 2*ti*nx + 2*((xi + nx) % nx) + 1

	# Stencil coefficients
	cuu = coefficients[0]
	cuv = coefficients[1]
	cvu = coefficients[2]
	cvv = coefficients[3]

	# Radii of stencils
	uu_imax = int((len(cuu)-1)/2)
	uv_imax = int((len(cuv)-1)/2)
	vu_imax = int((len(cvu)-1)/2)
	vv_imax = int((len(cvv)-1)/2)

	# Constants
	n = nx*nt*2
	
	# Number of nnzs in matrix	
	n_d = 2 * nx + ( (1 + len(cuu) + len(cuv)) + (1 + len(cvu) + len(cvv)) ) * nx *(nt-1)

	# Arrays for sparse matrix
	rowinds = np.zeros((n_d,), dtype='int32')
	colinds = np.zeros((n_d,), dtype='int32')
	data = np.zeros((n_d,), dtype='float')
	d_ind = 0
	b = np.zeros((n,))

	# Loop over points in time, then space
	for t in range(0,nt):
		for x in range(0,nx):

			# Set matrix to diagonal along boundary t=0, and set right hand
			# side vector
			if (t == 0):
				rowinds[d_ind] = get_u(t,x)
				colinds[d_ind] = get_u(t,x)
				data[d_ind] = 1.0
				d_ind += 1
				rowinds[d_ind] = get_v(t,x)
				colinds[d_ind] = get_v(t,x)
				data[d_ind] = 1.0
				d_ind += 1
				
				b[get_u(t,x)] = u0(x)
			else:
				# ----------------- row for u_i^(n+1) ----------------- #
				# u_i^(n+1)
				rowinds[d_ind] = get_u(t,x)
				colinds[d_ind] = get_u(t,x)
				data[d_ind] = 1.0
				d_ind += 1                

				# u_i^n connections
				for i in range(-uu_imax, uu_imax+1):
					rowinds[d_ind] = get_u(t,x)    
					colinds[d_ind] = get_u(t-1,x+i)
					data[d_ind] = -cuu[i + uu_imax]
					d_ind += 1

				# v_i^n connections
				for i in range(-uv_imax, uv_imax+1):
					rowinds[d_ind] = get_u(t,x)    
					colinds[d_ind] = get_v(t-1,x+i)
					data[d_ind] = -cuv[i + uv_imax]
					d_ind += 1


				# ----------------- row for v_i^(n+1) ----------------- #
				# v_i^(n+1)
				rowinds[d_ind] = get_v(t,x)
				colinds[d_ind] = get_v(t,x)
				data[d_ind] = 1.0
				d_ind += 1                

				# u_i^n connections
				for i in range(-vu_imax, vu_imax+1):
					rowinds[d_ind] = get_v(t,x)    
					colinds[d_ind] = get_u(t-1,x+i)
					data[d_ind] = -cvu[i + vu_imax]
					d_ind += 1

				# v_i^n connections
				for i in range(-vv_imax, vv_imax+1):
					rowinds[d_ind] = get_v(t,x)    
					colinds[d_ind] = get_v(t-1,x+i)
					data[d_ind] = -cvv[i + vv_imax]
					d_ind += 1    

	A = csr_matrix((data, (rowinds,colinds)), shape=(n,n))
	A.eliminate_zeros()
	A = A.tobsr(blocksize=[2,2])
	return A, b



# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Two-dimensional schemes


# For given problem parameters, compute maximum allowable time step size, dt.
def get_CFL_limit_2d(scheme, hx, cx, hy, cy):
	if scheme == "UW1":
		return 1/(cx/hx + cy/hy)
	elif scheme == "UW2":
		cx = cx/hx
		cy = cy/hy
		return 0.5*(cx + cy)*( np.sqrt(5 + 2*cx*cy/(cx+cy)**2) - 1 )/(cx**2 + cy**2)


# 1st-order scheme
# Coefficient ordering (array is oriented same as physical grid):
# 	Connection to u(xi,yj) is in the middle of the array
#	W (i < 0) to E (i > 0) horizontal connections correspond increasing column index
#	N (j > 0) to S (j < 0) vertical connections correspond to increasing row index
def get_UW1_2D(nt, dt, nx, hx, cx, ny, hy, cy, u0):
	Kx = cx * dt / hx
	Ky = cy * dt / hy

	# Stencil coefficients
	#				i-1					       i 		   i+1
	gu = np.array( [[0.0,                   Ky**2/dt,      0.0],  # j+1
					[Kx**2/dt, -2/dt*(Kx**2 + Ky**2), Kx**2/dt],  # j
					[0.0,                   Ky**2/dt,      0.0]]) # j-1
	
	#				i-1			  i 	i+1
	gv = np.array( [[0.0,        Ky/2,  0.0],  # j+1
					[Kx/2, -(Kx + Ky), Kx/2],  # j
					[0.0,        Ky/2,  0.0]]) # j-1

	cuu = dt/2*gu.copy()
	cuu[1,1] += 1
	cuv = dt/2*gv.copy()
	cuv[1,1] += dt

	cvu = gu.copy()
	cvv = gv.copy()
	cvv[1,1] += 1
	
	print(np.sum(cuu))
	print(np.sum(cuv))
	print(np.sum(cvu))
	print(np.sum(cvv))
	
	return get_2d_system(nt, nx, ny, [cuu, cuv, cvu, cvv], u0)
	
	
# 2nd-order scheme
# Coefficient ordering (array is oriented same as physical grid):
# 	Connection to u(xi,yj) is in the middle of the array
#	W (i < 0) to E (i > 0) horizontal connections correspond increasing column index
#	N (j > 0) to S (j < 0) vertical connections correspond to increasing row index
def get_UW2_2D(nt, dt, nx, hx, cx, ny, hy, cy, u0):
	Kx = cx * dt / hx
	Ky = cy * dt / hy

	# Stencil coefficients
	#				 i-2											i-1					       										i 		   								i+1		       i+2
	gu = np.array( [[0.0,                   						0.0, 							    					   Ky**3/(4*dt), 								0.0,           0.0],  # j+2
					[0.0,     				    	Kx*Ky/(4*dt)*(Kx+Ky), 							  Ky/(2*dt)*(-Kx*(Kx+Ky) + 2*Ky*(1-Ky)), 				Kx*Ky/(4*dt)*(Kx+Ky),          0.0],  # j+1
					[Kx**3/(4*dt),	 Kx/(2*dt)*(-Ky*(Kx+Ky)+2*Kx*(1-Kx)), -Kx**2/(2*dt)*(4-3*Kx) + Kx*Ky/dt*(Kx+Ky) - Ky**2/(2*dt)*(4-3*Ky), Kx/(2*dt)*(-Ky*(Kx+Ky)+2*Kx*(1-Kx)), Kx**3/(4*dt)],  # j
					[0.0,                   		Kx*Ky/(4*dt)*(Kx+Ky), 							  Ky/(2*dt)*(-Kx*(Kx+Ky) + 2*Ky*(1-Ky)), 				Kx*Ky/(4*dt)*(Kx+Ky),          0.0],  # j-1
					[0.0,                   						0.0, 													   Ky**3/(4*dt), 								 0.0,          0.0]]) # j-2
	
	#				 i-2			i-1					       		i 	 		 i+1     i+2
	gv = np.array( [[0.0,           0.0,						  -Ky/8, 		 0.0,   0.0],  # j+2
					[0.0,     		0.0, 					Ky/2*(1+Ky), 		 0.0,   0.0],  # j+1
					[-Kx/8,	Kx/2*(1+Kx), -Kx/4*(3+4*Kx) - Ky/4*(3+4*Ky), Kx/2*(1+Kx), -Kx/8],  # j
					[0.0,          	0.0, 					Ky/2*(1+Ky), 		 0.0, 	0.0],  # j-1
					[0.0,           0.0,						  -Ky/8, 		 0.0,   0.0]]) # j-2
	
	cuu = dt/2*gu.copy()
	cuu[2,2] += 1
	cuv = dt/2*gv.copy()
	cuv[2,2] += dt

	cvu = gu.copy()
	cvv = gv.copy()
	cvv[2,2] += 1
		
	return get_2d_system(nt, nx, ny, [cuu, cuv, cvu, cvv], u0)
	

# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Space-time system for arbitrary-order scheme, two spatial dimensions.
# Assume global ordering:
#	{ (x0,y0,t0), (x1,y0,t0), ..., (xn,y0,t0), (x0,y1,t0),...,
#	  (x0,yn,t0), ..., (x0,y0,t1),..., (xn,yn,tn) }
#
# Assumes that spatial stencils are centred.
def get_2d_system(nt, nx, ny, coefficients, u0):

	# Return global index for variable u at node (ti,xi,yi) 
	def get_u(ti, xi, yi):
		return 2*(ti*nx*ny + nx*((yi + ny) % ny) + (xi + nx) % nx)

	# Return global index for variable v at node (ti,xi,yi) 
	def get_v(ti, xi, yi):
		return 2*(ti*nx*ny + nx*((yi + ny) % ny) + (xi + nx) % nx) + 1

	# Stencil coefficients
	cuu = coefficients[0]
	cuv = coefficients[1]
	cvu = coefficients[2]
	cvv = coefficients[3]

	# Radii of stencils in horizontal and vertical directions
	uu_imax = int((cuu.shape[1]-1)/2) 
	uu_jmax = int((cuu.shape[0]-1)/2) 
	uv_imax = int((cuv.shape[1]-1)/2) 
	uv_jmax = int((cuv.shape[0]-1)/2) 
	vu_imax = int((cvu.shape[1]-1)/2) 
	vu_jmax = int((cvu.shape[0]-1)/2) 
	vv_imax = int((cvv.shape[1]-1)/2) 
	vv_jmax = int((cvv.shape[0]-1)/2) 

	# Constants
	n = nx*ny*nt*2

	# Number of nnzs in matrix
	n_d = nx*ny*( 2*nt + (nt-1)*(np.count_nonzero(cuu) + np.count_nonzero(cuv) + \
			np.count_nonzero(cvu) + np.count_nonzero(cvv)) ) 

	# Arrays for sparse matrix
	rowinds = np.zeros((n_d,), dtype='int32')
	colinds = np.zeros((n_d,), dtype='int32')
	data = np.zeros((n_d,), dtype='float')
	d_ind = 0
	b = np.zeros((n,))

	# Loop over points in time, then space
	for t in range(0,nt):
		for x in range(0,nx):
			for y in range(0,ny):
			
				# Set matrix to diagonal along boundary t=0, and set right hand
				# side vector
				if (t == 0):
					rowinds[d_ind] = get_u(t,x,y)
					colinds[d_ind] = get_u(t,x,y)
					data[d_ind] = 1.0
					d_ind += 1
					rowinds[d_ind] = get_v(t,x,y)
					colinds[d_ind] = get_v(t,x,y)
					data[d_ind] = 1.0
					d_ind += 1

					b[get_u(t,x,y)] = u0(x, y)
				else:
					# ----------------- row for u_i^(n+1) ----------------- #
					# u_ij^(n+1)
					rowinds[d_ind] = get_u(t,x,y)
					colinds[d_ind] = get_u(t,x,y)
					data[d_ind] = 1.0
					d_ind += 1                

					# u_ij^n connections
					for i in range(-uu_imax, uu_imax+1): # W to E
						for j in range(-uu_jmax, uu_jmax+1): # S to N
							if cuu[uu_jmax-j, i+uu_imax] != 0.0: # Connection to u^n_{x+i,y+j}
								rowinds[d_ind] = get_u(t,x,y)    
								colinds[d_ind] = get_u(t-1,x+i,y+j) 
								data[d_ind] = -cuu[uu_jmax-j, i+uu_imax]
								d_ind += 1

					# v_ij^n connections
					for i in range(-uv_imax, uv_imax+1): # W to E
						for j in range(-uv_jmax, uv_jmax+1): # S to N
							if cuv[uv_jmax-j, i+uv_imax] != 0.0: # Connection to v^n_{x+i,y+j}
								rowinds[d_ind] = get_u(t,x,y)    
								colinds[d_ind] = get_v(t-1,x+i,y+j) 
								data[d_ind] = -cuv[uv_jmax-j, i+uv_imax]
								d_ind += 1
								
								
					# ----------------- row for v_i^(n+1) ----------------- #
					# v_ij^(n+1)
					rowinds[d_ind] = get_v(t,x,y)
					colinds[d_ind] = get_v(t,x,y)
					data[d_ind] = 1.0
					d_ind += 1
					
					# u_ij^n connections
					for i in range(-vu_imax, vu_imax+1): # W to E
						for j in range(-vu_jmax, vu_jmax+1): # S to N
							if cvu[vu_jmax-j, i+vu_imax] != 0.0: # Connection to u^n_{x+i,y+j}
								rowinds[d_ind] = get_v(t,x,y)    
								colinds[d_ind] = get_u(t-1,x+i,y+j) 
								data[d_ind] = -cvu[vu_jmax-j, i+vu_imax]
								d_ind += 1

					# v_ij^n connections
					for i in range(-vv_imax, vv_imax+1): # W to E
						for j in range(-vv_jmax, vv_jmax+1): # S to N
							if cvv[vv_jmax-j, i+vv_imax] != 0.0: # Connection to v^n_{x+i,y+j}
								rowinds[d_ind] = get_v(t,x,y)    
								colinds[d_ind] = get_v(t-1,x+i,y+j) 
								data[d_ind] = -cvv[vv_jmax-j, i+vv_imax]
								d_ind += 1

	A = csr_matrix((data, (rowinds,colinds)), shape=(n,n))
	A.eliminate_zeros()
	A = A.tobsr(blocksize=[2,2])
	
	return A, b
