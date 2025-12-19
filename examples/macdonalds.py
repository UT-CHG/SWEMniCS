from swemnics.problems import TidalProblem, FlumeExperiment
from swemnics import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit

'''
Based on case from paper:
Towards transient experimental water surfaces: A new benchmark dataset
for 2D shallow water solvers
'''

start = timeit.default_timer()



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#paramterize by input
h5_file_path = 'my_data'

dt = 0.1/5.0#0.1
t = 0
t_f = 30.0#10.0
nt = int(np.ceil(t_f/dt))
mannings_n = 0.01
print('nmber of time steps',nt)
#friction law either quadratic or linear
fric_law = 'mannings'
#choose solution variable, either h or eta or flux
sol_var = 'h'
cm_to_m = .01
# width of rectangle
r_width = 16.3*cm_to_m
#height of rectangle object
r_height = 8.0*cm_to_m
L = 6.0078
#original expirement value
#H = 24.0*cm_to_m
#extended domain
H = r_height*11.0

y_coord = 0.12
#depth on left boundary
boundary_depth = 28.0/100.0
# take m3/s and convert to m2/s by dividing by width of inflow
# channel width = .24 m 
# exp 1: inflow = 5.05 m3/h
inflow_rate = 5.05/(60*60*H)
# exp 2: inflow = 9.01 m3/h
#inflow_rate = 9.01/(60*60*H)
# exp 3: inflow = 12.01 m3/h
#inflow_rate = 12.01/(60*60*H)
prob = FlumeExperiment(dt=dt,nt=nt,friction_law=fric_law,
						  solution_var=sol_var,wd_alpha=0.001,wd=True,
						  TAU=mannings_n, boundary_flux=inflow_rate, h_b_val=boundary_depth,
						  xdmf_file="data/Flume/mesh.xdmf",
						  xdmf_facet_file="data/Flume/facet_mesh.xdmf")
'''
prob = TidalProblem(
    nx=20,
    ny=5,
    dt=3600,
    nt=24*7,
    friction_law="mannings",
    solution_var="h",
)
'''
p_degree = [1,1]
rel_toleran=1e-5
abs_toleran=1e-6
max_iter=10
relax_param = 1.0
#time series output
#generating grid points
#dont cover whole grid
#instead do roughly 10d behind, 5d in front
# 10d is roughly 1.6 m 
# spacing should be every .01 m which is finest resolution
npx = 601
npy = 89
npoints = npx*npy
eps = 1e-7
stations = np.zeros((npx*npy,3))
just_x = np.linspace(0.0+eps,6.00,npx)
just_y = np.linspace(0+eps,H-eps,npy)
stations[:,0] = np.tile(just_x,npy)
stations[:,1] = np.repeat(just_y,npx)
#nstat = 12
#stations = np.hstack(( np.linspace(0.0,L,nstat),y_coord*np.ones(nstat),np.zeros(nstat)))
#create solver object

#cg
theta=1
#solver = Solvers.CGImplicit(prob,theta)
#supg, not working yet with wd
#solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
#dg DGImplicit
#solver = Solvers.DGImplicit(prob,theta,p_degree=p_degree,make_tangent=False, get_station_h=True)
#dg non conservative
#solver = Solvers.DGImplicitNonConservative(prob,theta,p_degree=p_degree)
# linearized special
solver = Solvers.Linearized_DG(prob,theta,p_degree=p_degree,make_tangent=False, get_station_h=True)

params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "bjacobi", "ksp_ErrorIfNotConverged": False}#,"pc_factor_mat_solver_type":"mumps"}
name='MacDonald'
solver.time_loop(solver_parameters=params,stations=stations,plot_every=1,plot_name=name)