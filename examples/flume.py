from swemnics.problems import FlumeExperiment
from swemnics import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit
import h5py

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
#inflow_rate = 5.05/(60*60*H)
# exp 2: inflow = 9.01 m3/h
#inflow_rate = 9.01/(60*60*H)
# exp 3: inflow = 12.01 m3/h
inflow_rate = 12.01/(60*60*H)


prob = FlumeExperiment(dt=dt,nt=nt,friction_law=fric_law,
						  solution_var=sol_var,wd_alpha=0.001,wd=True,
						  TAU=mannings_n, boundary_flux=inflow_rate, h_b_val=boundary_depth,
						  xdmf_file="data/Flume/mesh.xdmf",
						  xdmf_facet_file="data/Flume/facet_mesh.xdmf")
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
solver = Solvers.DGImplicit(prob,theta,p_degree=p_degree,make_tangent=False, get_station_h=True)
#dg non conservative
#solver = Solvers.DGImplicitNonConservative(prob,theta,p_degree=p_degree)
params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "bjacobi", "ksp_ErrorIfNotConverged": False}#,"pc_factor_mat_solver_type":"mumps"}
name='Flume'
solver.time_loop(solver_parameters=params,stations=stations,plot_every=10,plot_name=name)

#solver.solve()
#prob.plot_solution(solver.u.sub(0),'Single_time_step')
#print(solver.station_data.shape)
if rank ==0:
	#note that station data is array with shape nt x nstattion x 3 (h,u,v)
	plt.plot(np.linspace(0,t_f/(60*60*24),nt+1), solver.vals[:nt+1,-1,0], "k", linewidth=2, label="h at 800 m")
	#plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
	plt.grid(True)
	plt.xlabel("t(day)")
	plt.title('WDTidal Height for DG Scheme')
	plt.savefig("wd_flume_height_DG.png")

	plt.close()

	plt.plot(np.linspace(0,t_f/(60*60*24),nt+1), solver.vals[:nt+1,-1,1], "k", linewidth=2, label="u at 800 m")
	#plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
	plt.grid(True)
	plt.xlabel("t(day)")
	plt.title('WD Tidal Velocity for DG Scheme')
	plt.savefig("wd_flume_velocity_DG.png")

	#save to each solution to h5
	print(solver.vals.shape)
	print(stations.shape)
	with h5py.File(h5_file_path+"_h.h5", 'w') as f:
    	# Create a dataset named 'my_array_dataset' and store data_array in it
		# there are gaps in the data
		temp_arr = np.full((nt+1,npoints),fill_value=np.nan)
		temp_arr[:,solver.inds] = solver.vals[:,:,0]
		temp_arr = temp_arr.reshape((nt+1,npy,npx))
		f.create_dataset('h', data=temp_arr)
	with h5py.File(h5_file_path+"_u.h5", 'w') as f:
    	# Create a dataset named 'my_array_dataset' and store data_array in it
		# there are gaps in the data
		temp_arr = np.full((nt+1,npoints),fill_value=np.nan)
		temp_arr[:,solver.inds] = solver.vals[:,:,1]
		temp_arr = temp_arr.reshape((nt+1,npy,npx))
		f.create_dataset('u', data=temp_arr)
		#print(np.argwhere(np.isnan(temp_arr)))
	with h5py.File(h5_file_path+"_v.h5", 'w') as f:
    	# Create a dataset named 'my_array_dataset' and store data_array in it
		# there are gaps in the data
		temp_arr = np.full((nt+1,npoints),fill_value=np.nan)
		temp_arr[:,solver.inds] = solver.vals[:,:,2]
		temp_arr = temp_arr.reshape((nt+1,npy,npx))
		f.create_dataset('v', data=temp_arr)
		#print(np.argwhere(np.isnan(temp_arr)))
	with h5py.File(h5_file_path+"_bathy.h5", "w") as f:
		temp_arr = np.full(npoints,fill_value=np.nan)
		temp_arr[solver.inds] = solver.station_bathy.flatten()
		temp_arr = temp_arr.reshape((npy,npx))
		f.create_dataset('bathy', data=temp_arr)
		#print(np.argwhere(np.isnan(temp_arr)))

	#np.savetxt(f"{name}_p1_wse.csv", solver.vals[:,:,0], delimiter=",")
	#np.savetxt(f"{name}_p1_xvel.csv", solver.vals[:,:,1], delimiter=",")
	#np.savetxt(f"{name}_p1_yvel.csv", solver.vals[:,:,2], delimiter=",")

#Your statements here

stop = timeit.default_timer()

print('Time: ', stop - start)
