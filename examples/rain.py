from swemnics.problems import RainProblem
from swemnics import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit
import argparse as ap
import os

def run_experiment(name, outdir=None, **kwargs):
    start = timeit.default_timer()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nx = ny = 10
    
    dt = 5.0
    t = 0
    t_f = 48*60*60

    nt = int(np.ceil(t_f/dt))
    print('nmber of time steps',nt)
    #friction law either quadratic or linear
    fric_law = 'linear'
    #choose solution variable, either h or eta or flux
    sol_var = 'h'

    prob = RainProblem(dt = dt, nt = nt,
        spherical=False
    )
    p_degree = [1,1]
    rel_toleran=1e-12
    abs_toleran=1e-13
    max_iter=10
    relax_param = 1.0
    #time series output
    #time series output
    nx=100
    stations = np.zeros((nx,3))
    stations[:,0] = np.linspace(0,1000,nx)
    stations[:,1] = 450
    #create solver object
    
    #cg
    theta=1
    if name == "cg":
        solver = Solvers.CGImplicit(prob,theta, **kwargs)
    #supg
    elif name == "supg":
        solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree, **kwargs)
    #dg DGImplicit
    elif name == "dg":
        solver = Solvers.DGImplicit(prob,theta,p_degree=p_degree, **kwargs)
    #dgcg
    elif name == "dgcg":
        solver = Solvers.DGCGImplicit(prob,theta,p_degree=p_degree, **kwargs)
    else:
        raise ValueError(f"Unrecognized solver '{name}'")
    
    name = name.upper()
    params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "ilu"}#,"pc_factor_mat_solver_type":"mumps"}
    solver.time_loop(solver_parameters=params,stations=stations,plot_every=1200,plot_name='rain_test_'+name)
    
    #solver.solve()
    #prob.plot_solution(solver.u.sub(0),'Single_time_step')
    #print(solver.station_data.shape)
    #save array for post processing
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)        
    outdir = "" if outdir is None else outdir+"/"
    np.savetxt(f"{outdir}{name}_p1_stations_h.csv", solver.vals[:,:,0], delimiter=",")
    np.savetxt(f"{outdir}{name}_p1_stations_xvel.csv", solver.vals[:,:,1], delimiter=",")
    np.savetxt(f"{outdir}{name}_p1_stations_yvel.csv", solver.vals[:,:,2], delimiter=",")
    if rank ==0:
        plt_nums = [0,nt]
    	#note that station data is array with shape nt x nstattion x 3 (h,u,v)
        for a in plt_nums:
            if a > nt: break
            t=a*dt
            plt.plot(np.linspace(0,1000,nx), solver.vals[a,:,0], "--", linewidth=2, label="h at "+str(int(t*10)/10))
        #plt.plot(np.linspace(0,1000,100), solver.vals[1,:,0], linewidth=2, label="h at 100")
        #plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
        plt.grid(True)
        plt.xlabel("x(m)")
        plt.ylabel('surface elevation(m)')
        plt.title(f'Surface Elevation for {name} Scheme')
        plt.legend()
        plt.savefig(f"{outdir}rain_height_{name}_order1_dt.png")
        plt.close()
        #plt.plot(np.linspace(0,t_f/(60*60*24),nt+1), solver.vals[:nt+1,0,1], "k", linewidth=2, label="u at 800 m")
        #plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
        #plt.grid(True)
        #plt.xlabel("t(day)")
        #lt.title(f'Tidal Velocity for {name} Scheme')
        #plt.savefig(f"dam_velocity_{name}.png")
    
    #Your statements here
    
    stop = timeit.default_timer()
    
    print('Time: ', stop - start)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "solver",
        choices=["cg", "supg", "dg", "dgcg"]
    )
    args = parser.parse_args() 
    run_experiment(args.solver)
