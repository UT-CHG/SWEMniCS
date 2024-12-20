from swemnics import solvers as Solvers
from swemnics.adcirc_problem import ADCIRCProblem
from mpi4py import MPI
import numpy as np
from swemnics.forcing import GriddedForcing
from swemnics.constants import R
import time
import argparse as ap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def bathy_hack_func(prob, depth):
    V = depth.function_space
    coords = V.tabulate_dof_coordinates()
    coords[:,0] /= np.cos(np.deg2rad(35))
    coords = np.rad2deg(coords[:,:2] / R)
    lon, lat = coords[:, 0], coords[:, 1]
    mask = (lon > -91) | (lat < 25)
    arr = depth.x.array[:]
    depth.x.array[(arr < 5) & mask] = 5


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--dt", type=float, default=3600)
    parser.add_argument("--solver", choices=['dg', 'supg'], default='dg')
    parser.add_argument("--no-plot", action="store_true", default=False)
    parser.add_argument("--stations", default=None)
    parser.add_argument("--friction", default='quadratic', choices=['linear', 'quadratic', 'nolibf2', 'mannings'])
    parser.add_argument("--mesh", default="small_gulf")
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()


    dt = args.dt
    t = 0
    t_f = 8.75*24*3600
    nt = int((t_f-t)/dt)
    is_spherical=True
    bath_adjust=5
    args.tide_only=True
    # The forcing file for Ike cannot be included due to licensing restrictions
    # so this example is purely tidal
    forcing=None

    prob = ADCIRCProblem(adios_file="data/"+args.mesh, spherical=is_spherical,
        solution_var='h',
        friction_law=args.friction,
        forcing=forcing,
        dramp=.25,
        dt=dt, bathy_adjustment=bath_adjust, nt=nt,
                        sea_surface_height=.22,
                        t=t,
                        bathy_hack_func = bathy_hack_func,
                        min_bathy=0.3
                        )
    #prob.plot_mesh()
    p_degree = 1
    #rel_toleran=1e-6
    #abs_toleran=1e-6
    rel_toleran=abs_toleran=1e-5
    max_iter=20
    relax_param = 1
    theta=1
    
    #first number is continuity, second is momentum
    p_degree = [1,1]
    params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "bjacobi"}#,"pc_factor_mat_solver_type":"mumps"}
    params['ksp_rtol'] = 1e-6
    params['ksp_atol'] = 1e-6
    
    #supg
    if args.solver=='supg':
      params['max_it'] = 20
      params['ksp_max_it'] = 5000
      solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
    #dg DGImplicit
    elif args.solver == 'dg':
      solver = Solvers.DGImplicit(prob,theta, p_degree=p_degree, cuda=args.cuda)
    else:
      raise ValueError(f"Unrecognized solver {args.solver}")
    #Dg continuity CG momentum
    #solver = Solvers.DGCGImplicit(prob,theta, p_degree=p_degree)
    
    #time and print
    plot_name = f'ike_{args.solver}_{args.friction}_dt{int(args.dt)}_nprocs{size}'
    if args.tide_only: plot_name += '_tide_only'
    rank = MPI.COMM_WORLD.Get_rank()
    if args.stations is not None:
      import pandas as pd
      stations_df = pd.read_csv(args.stations)
      coords = stations_df[['lon', 'lat']].values
      stations = prob.apply_projection(coords)
    else:
      stations = []

    start_time = time.time()
    solver.time_loop(solver_parameters=params,
                        plot_every=max(1, int(3600/dt)) if not args.no_plot else 10**18,
                        plot_name=plot_name,
                        stations=stations
                    )
    if rank == 0:
      elapsed = time.time() - start_time
      print("---------Simulation finished with run time %s seconds -------------" % (elapsed) )
      station_data = solver.vals
      station_index = solver.inds
      import os
      import json
      if not os.path.exists(plot_name):
          os.makedirs(plot_name, exist_ok=True)

      #only save runtime if plotting is turned off
      if args.no_plot:
          with open(plot_name+"/results.json", "w") as fp:
              info = {**vars(args)}
              info['runtime'] = elapsed
              info['nprocs'] = size
              json.dump(info, fp)

      if args.stations is not None:
        inds = np.arange(station_data.size)
        t_inds = np.floor(inds/station_data[0].size).astype(int)
        s_inds = station_index[np.floor((inds % station_data[0].size) / 3).astype(int)[::3]]
        
        df = pd.DataFrame({
            'time': t_inds[::3] * dt,
            'zeta': station_data[..., 0].flatten(),
            'u-vel': station_data[..., 1].flatten(),
            'v-vel': station_data[..., 2].flatten()
        })
        for c in stations_df.columns:
            df[c] = stations_df[c].values[s_inds]
        df.to_csv(plot_name+'/stations.csv', index=False)


