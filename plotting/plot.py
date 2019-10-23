import numpy as np

import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv

import pdb
from numpy.linalg import norm
from scipy.sparse import load_npz

# Plot solution to 1D  test  problems  for advection

# Sit there here to enable latex-style font in plots...
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.usetex'] = True

# Run me like << python plot.py <<path to file with solution information>>
# File should contain a list of (at least) the following parameters:
# P
# nt
# dt
# s
# problemID
# space_dim
# spaceParallel
# nx

# if len(argv) > 1:
#     filename = argv[1]
# else:
#     raise ValueError("A filename must be passed through command line!")

filename = "../ST_class/data/X_FD.txt"

# Read data in and store in dictionary
params = {}
with open(filename) as f:
    for line in f:
       (key, val) = line.split()
       params[key] = val
       
# Type cast the parameters from strings into their original types
params["P"]               = int(params["P"])
params["nt"]              = int(params["nt"])
params["s"]               = int(params["s"])
params["dt"]              = float(params["dt"])
params["problemID"]       = int(params["problemID"])
params["nx"]              = int(params["nx"])
params["space_dim"]       = int(params["space_dim"])
params["spatialParallel"] = int(params["spatialParallel"])

# Total number of DOFS in space
if params["space_dim"] == 1:
    NX = params["nx"] 
elif params["space_dim"] == 2:
    NX = params["nx"] ** 2


### ----------------------------------------------------------------------------------- ###
### --- NO SPATIAL PARALLELISM: Work out which processor uT lives on and extract it --- ###
if not params["spatialParallel"]:
    DOFsPerProc = int((params["s"] * params["nt"]) / params["P"]) # Number of temporal variables per proc
    PuT         = int(np.floor( (params["s"] * (params["nt"]-1)) / DOFsPerProc )) # Index of proc that owns uT
    PuT_DOF0Ind = PuT * DOFsPerProc # Global index of first variable on this proc
    PuT_uTInd   = (params["s"] * (params["nt"]-1)) - PuT_DOF0Ind # Local index of uT on its proc
    
    # Filename for data output by processor output processor. Assumes format is <<filename>>.<<index of proc using 5 digits>>
    Ufilename  = filename + "." + "0" * (5-len(str(PuT))) + str(PuT)

    # Read all data from this proc
    with open(Ufilename) as f:
        dims = f.readline()
    dims.split(" ")
    dims = [int(x) for x in dims.split()] 
    # Get data from lines > 0
    uT_dense = np.loadtxt(Ufilename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need
        
    # Extract uT from other junk    
    uT = np.zeros(NX)
    uT[:] = uT_dense[PuT_uTInd*NX:(PuT_uTInd+1)*NX]


### ---------------------------------------------------------------------------------------- ###
### --- SPATIAL PARALLELISM: Work out which processors uT lives on extract it from them ---  ###
# Note that ordering is preserved...
else:
    params["spatial_Np_x"]    = int(params["spatial_Np_x"])
    
    # Index of proc holding first component of uT
    PuT0 = (params["nt"]-1) * params["s"] * params["spatial_Np_x"]     
    
    # Get names of all procs holding uT data
    PuT = []
    for P in range(PuT0, PuT0 + params["spatial_Np_x"]):
        PuT.append(filename + "." + "0" * (5-len(str(P))) + str(P))
    
    uT = np.zeros(NX)
    
    DOFsPerProc = int(NX/params["spatial_Np_x"])
    
    for count, Ufilename in enumerate(PuT):
        #print(count, NX)
        # Read all data from the proc
        with open(Ufilename) as f:
            dims = f.readline()
        dims.split(" ")
        dims = [int(x) for x in dims.split()] 
        # Get data from lines > 0
        uT[count*DOFsPerProc:(count+1)*DOFsPerProc] = np.loadtxt(Ufilename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need
    
    
    

### --------------------------------------------- ###
### --- PLOTTING: uT against exact soliution ---  ###
### --------------------------------------------- ###

###  1D
if params["space_dim"] == 1:
    nx = NX
    x = np.linspace(-1, 1, nx+1)
    x = x[:-1] # nx points in [-1, 1)
    T = params["dt"]  * (params["nt"] - 1)

    # The exact solutions for the test problems
    def uexact(x,t):
        if params["problemID"] == 1:
            temp = np.mod(x + 1  - t, 2) - 1
            return np.cos(np.pi*temp) ** 4
        elif (params["problemID"] == 2) or (params["problemID"] == 3):    
            return np.cos(np.pi*(x - t)) * np.exp(np.cos(t))/np.exp(1)

    uT_exact = np.zeros(nx)
    for i in range(0,nx):
        uT_exact[i] = uexact(x[i],T)

    # Compare uT against the exact solution
    #print("nx = {}, |uNum - uExact| = {:.4e}".format(nx, np.linalg.norm(uT_exact - uT, np.inf)))
    print("(nt,nx) = ({},{}), |uNum - uExact| = {:.4e}".format(params["nt"], nx, np.sqrt(2/nx) * np.linalg.norm(uT_exact - uT, 2)))

    plt.plot(x, uT_exact, linestyle = "--", marker = "o", markerfacecolor = "none", color = "r", label = "$u_{{\\rm{exact}}}$")
    plt.plot(x, uT, linestyle = "--", marker = "x", color = "b", label = "$u_{{\\rm{num}}}$")
    
    fs = 18
    plt.legend(fontsize = fs)
    plt.title("$\\rm{{P}}_{{\\rm{{ID}}}}$={}:\t(RK, U-order, $n_x$, $T_{{\\rm{{f}}}}$)=({}, {}, {}, {:.2f})".format(params["problemID"], params["timeDisc"], params["space_order"], nx, T), fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.show()
    
    
### --------------------------------------------------- ###
### ----------------------- 2D  ----------------------- ###
### --------------------------------------------------- ###
if params["space_dim"] == 2:
    nx = params["nx"]
    ny = nx
    x = np.linspace(-1, 1, nx+1)
    y = np.linspace(-1, 1, ny+1)
    x = x[:-1] # nx points in [-1, 1)
    y = y[:-1] # ny points in [-1, 1)
    [X, Y] = np.meshgrid(x, y)
    T = params["dt"]  * (params["nt"] - 1)

    uT = uT.reshape(ny, nx)

    def u0(x,y):
        #return np.cos(np.pi*x) ** 4 * np.cos(np.pi*y) ** 4
        return np.cos(np.pi*x) ** 4 * np.sin(np.pi*y) ** 2
        #return np.cos(np.pi*x)

    # The exact solutions for the test problems
    def uexact(x, y ,t):
        if params["problemID"] == 1:
            tempx = np.mod(x + 1  - t, 2) - 1
            tempy = np.mod(y + 1  - t, 2) - 1
            return u0(tempx, tempy)

    uT_exact = np.zeros((nx, ny))
    for j in range(0,ny):
        for i in range(0,nx):
            #uT_exact[i,j] = uexact(x[i],y[j],T)
            uT_exact[j,i] = uexact(x[i],y[j],T)
            

    # print(uT)
    # 
    # print("\n\noooooh\n\n")
    # 
    # print(uT_exact)

    # Compare uT against the exact solution
    print("nx = {}, |uNum - uExact| = {:.4e}".format(nx, np.linalg.norm(uT_exact - uT, np.inf)))
    #print("(nt,nx) = ({},{}), |uNum - uExact| = {:.4e}".format(params["nt"], nx, np.sqrt(2/nx) * np.linalg.norm(uT_exact - uT, 2)))


    fs = 18
    cmap = plt.cm.get_cmap("coolwarm")
    
    fig = plt.figure(1)
    ax = fig.gca(projection='3d') 
    surf = ax.plot_surface(X, Y, uT, cmap = cmap)
    #plt.title("$\\rm{{P}}_{{\\rm{{ID}}}}$={}:\t(RK, U-order, $n_x$, $T_{{\\rm{{f}}}}$)=({}, {}, {}, {:.2f})".format(params["problemID"], params["timeDisc"], params["space_order"], nx, T), fontsize = fs)
    plt.title("uNum", fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$y$", fontsize = fs)
    
    
    fig = plt.figure(2)
    ax = fig.gca(projection='3d') 
    surf = ax.plot_surface(X, Y, uT_exact, cmap = cmap)
    plt.title("uTexact", fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$y$", fontsize = fs)
    plt.show()    
    
    # plt.title("UW2-2D: u(x,y, t = {:.2f})".format(t[-1]), fontsize = 15)
    # plt.xlabel("x", fontsize = 15)
    # plt.ylabel("y", fontsize = 15)


