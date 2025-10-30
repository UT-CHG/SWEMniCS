from scipy.io import loadmat
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

mat_file_path = "/Users/markloveland/SWEMniCS/examples/data/Flume/random_field_A.mat"
mat_data = loadmat(mat_file_path)
variable_name = "A_random"  # Replace with the actual variable name from your .mat file
data = mat_data[variable_name].T
x_coords = mat_data["x_data"][:,0]
nsample,npoints = data.shape
print(f"n samples {nsample}, n grid points {npoints}")
'''
for i in range(nsample):
    plt.plot(x_coords, data[i])
plt.show()
'''

#in numpy array, how to make readable by fenicsx
import dolfinx
from mpi4py import MPI
import numpy as np
import dolfinx.fem.petsc as petsc
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)

dof_coordinates = V.tabulate_dof_coordinates()[boundary_dofs]
print(dof_coordinates)
u_bc = dolfinx.fem.Function(V)

# make f an interpolation function from the data

f_cubic= CubicSpline(x_coords[:],data[0],bc_type='natural')


def f(x):
    print(x)
    # Look-up x,y,z coordinates, preferrably in a vectorized fashion
    return f_cubic(x[0])

u_bc.interpolate(f)

with dolfinx.io.VTXWriter(mesh.comm, "u.bp", u_bc) as vtx:
    vtx.write(0.0)