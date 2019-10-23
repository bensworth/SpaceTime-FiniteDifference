from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import matplotlib.pyplot as plt
import pdb
from numpy.linalg import norm
from scipy.sparse import load_npz

###--- Make a heat map of the matrix output from HYPRE
A_hypre_path = "../ST_class/A_FD.mm.00000" # The filename of the matrix

# Get row/col numbers from first line of HYPRE out file.
with open(A_hypre_path) as f:
    dims = f.readline()
dims.split(" ")
dims = [int(x) for x in dims.split()] 
# Get mm data from lines > 0
dat = np.loadtxt(A_hypre_path, skiprows = 1)

A_hypre = csr_matrix((dat[:,2],(dat[:,0],dat[:,1])),shape=(dims[1]+1, dims[3]+1))
A_hypre.data[np.abs(A_hypre.data)<1e-15] = 0.0
A_hypre.eliminate_zeros()
print(A_hypre.shape, A_hypre.nnz)

plt.figure(1)
#plt.spy(A_hypre)
#A_hypre = np.abs(A_hypre)
plt.imshow(A_hypre.todense())
plt.title("HYPRE")
plt.colorbar()
plt.show()



## Check that the space-time matrix is the same if it's assembled across one or multiple procs. 
## Can just copy and paste the files from the individual processors into one file.
# A_hypre_path = ["n1.mm", "n2.mm"] # File names of matrices assembled on 1 and multiple procs
# A_hypre = []
# 
# # Get row/col numbers from first line of HYPRE out file.
# for count, matID in enumerate(A_hypre_path):
#     print(count)
#     with open(matID) as f:
#         dims = f.readline()
#     dims.split(" ")
#     dims = [int(x) for x in dims.split()] 
#     # Get mm data from lines > 0
#     dat = np.loadtxt(A_hypre_path[count], skiprows = 1)
# 
#     A_hypre.append(csr_matrix((dat[:,2],(dat[:,0],dat[:,1])),shape=(dims[1]+1, dims[3]+1)))
#     print(A_hypre[count].shape)
# 
# # Compute the difference between the two matricxes and report its norm    
# D = A_hypre[0]  - A_hypre[1]    
# print("norm(A_n1 - A_n2) = {:.2e}\n".format(norm(D.data)))
# 


