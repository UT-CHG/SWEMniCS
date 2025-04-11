from swemnics.problems import SlopedBeachProblem
from swemnics import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit
from dolfinx import fem as fe

start = timeit.default_timer()



comm = MPI.COMM_WORLD
rank = comm.Get_rank()



dt = 100
t = 0
t_f = 7*24*60*60
nt = int(np.ceil(t_f/dt))
print('nmber of time steps',nt)
#friction law either quadratic or linear
fric_law = 'mannings'
#choose solution variable, either h or eta or flux
sol_var = 'h'
prob = SlopedBeachProblem(dt=dt,nt=nt,friction_law=fric_law,solution_var=sol_var,wd_alpha=0.36,wd=True)


p_degree = [1,1]
rel_toleran=1e-5
abs_toleran=1e-6
max_iter=10
relax_param = 1.0
#time series output
stations = np.array([[0.0, 3650.0, 0.0],
 [1000.0, 3650.0, 0.0],
 [2000.0, 3650.0, 0.0],
 [3000.0, 3650.0, 0.0],
 [4000.0, 3650.0, 0.0],
 [5000.0, 3650.0, 0.0],
 [6000.0, 3650.0, 0.0],
 [7000.0, 3650.0, 0.0],
 [8000.0, 3650.0, 0.0],
 [9000.0, 3650.0, 0.0],
 [10000.0, 3650.0, 0.0],
 [11000.0, 3650.0, 0.0],
 [12000.0, 3650.0, 0.0],
 [13000.0, 3650.0, 0.0],
 [13500.0, 3650.0, 0.0],
 [13800.0, 3650.0, 0.0]])
#create solver object

#cg
theta=1
#solver = Solvers.CGImplicit(prob,theta)
#supg, not working yet with wd
#solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
#dg DGImplicit
solver = Solvers.DGImplicit(prob,theta,p_degree=p_degree)
params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "ilu"}#,"pc_factor_mat_solver_type":"mumps"}
name='Hotstart'



#overwrite start time and time stepping
#specify hotstart time (important for tides n stuff)
#in seconds
t0 = 8*60*60
solver.problem.t = t0

#need to be able to input initial condition
#full function space
V = solver.V
#full initial condition
u_0 = solver.u_n
#just water depth
h_0 = u_0.sub(0)
#bathymetry
h_b = solver.problem.h_b
#function space of just h
V_scalar = h_b._V


#pass initial conditon to solver
final_u,_,_= solver.time_loop(solver_parameters=params,stations=stations,plot_every=60,plot_name=name,u_0=u_0)

#after run, get state and modify
#create a function to interpolate a new initial condition to
wse_0 = fe.Function(V_scalar)

#interpolate the final wse first then modify some values
wse_0.interpolate(
	fe.Expression(final_u.sub(0) - h_b, 
				V.sub(0).element.interpolation_points())
                )
#add noise to this just to test, this will be where we put corrections in
wse_0.x.array[:] = 0.5
#recall wse is h-h_b
#h_0=h_b+wse_0
#use this to 
#modify new intiial condition
h_0.interpolate(
	fe.Expression(wse_0 + h_b, 
				V.sub(0).element.interpolation_points())
                )
#call solver again with new state
final_u,_,_= solver.time_loop(solver_parameters=params,stations=stations,plot_every=60,plot_name=name,u_0=u_0)



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
	np.savetxt(f"{name}_p1_wse.csv", solver.vals[:,:,0], delimiter=",")
	np.savetxt(f"{name}_p1_xvel.csv", solver.vals[:,:,1], delimiter=",")
	np.savetxt(f"{name}_p1_yvel.csv", solver.vals[:,:,2], delimiter=",")

#Your statements here

stop = timeit.default_timer()

print('Time: ', stop - start)
