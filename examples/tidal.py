from swemnics.problems import TidalProblem
from swemnics import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit

start = timeit.default_timer()



comm = MPI.COMM_WORLD
rank = comm.Get_rank()


nx = 20
ny = 5
dt = 3600
t = 0
t_f = 7*24*60*60#24*7
nt = int(np.ceil(t_f/dt))
print('nmber of time steps',nt)
#friction law either quadratic or linear
fric_law = 'mannings'
#choose solution variable, either h or eta or flux
sol_var = 'h'


prob = TidalProblem(nx=ny, ny=ny,dt=dt,nt=nt,friction_law=fric_law,solution_var=sol_var)
p_degree = [1,1]
rel_toleran=1e-5
abs_toleran=1e-6
max_iter=10
relax_param = 1.0
#time series output
stations = np.array([[800.5,1000.5,0.0]])
#create solver object

#cg
theta=1
#solver = Solvers.CGImplicit(prob,theta)
#supg
#solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
#dg DGImplicit
#solver = Solvers.DGImplicit(prob,theta,p_degree=p_degree)
#dg DGImplicit
solver = Solvers.DGImplicitNonConservative(prob,theta,p_degree=p_degree)
params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "ilu"}#,"pc_factor_mat_solver_type":"mumps"}
solver.time_loop(solver_parameters=params,stations=stations,plot_every=1,plot_name='SUPG_tide')

#solver.solve()
#prob.plot_solution(solver.u.sub(0),'Single_time_step')
#print(solver.station_data.shape)
if rank ==0:
	#note that station data is array with shape nt x nstattion x 3 (h,u,v)
	plt.plot(np.linspace(0,t_f/(60*60*24),nt+1), solver.vals[:nt+1,0,0], "k", linewidth=2, label="h at 800 m")
	#plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
	plt.grid(True)
	plt.xlabel("t(day)")
	plt.title('Tidal Height for SUPG Scheme')
	plt.savefig("tidal_height_SUPG.png")

	plt.close()

	plt.plot(np.linspace(0,t_f/(60*60*24),nt+1), solver.vals[:nt+1,0,1], "k", linewidth=2, label="u at 800 m")
	#plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
	plt.grid(True)
	plt.xlabel("t(day)")
	plt.title('Tidal Velocity for SUPG Scheme')
	plt.savefig("tidal_velocity_SUPG.png")

#Your statements here

stop = timeit.default_timer()

print('Time: ', stop - start)
