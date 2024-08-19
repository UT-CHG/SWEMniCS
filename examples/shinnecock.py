from swemnics import solvers as Solvers
from swemnics.adcirc_problem import ADCIRCProblem
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from swemnics.forcing import GriddedForcing
from swemnics.constants import R
import time
import argparse as ap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--solver", choices=['dg', 'supg'], default='supg')
    parser.add_argument("--dt", type=float, default=600)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()
    print('Running shinnecock')
    dt = args.dt
    t_f = 5*24*3600-3600
    nt = int(t_f/dt)
    is_spherical=True
    wd = args.alpha is not None
    bath_adjust = 0 if wd else 4.0
    dramp = 2.0
    prob = ADCIRCProblem(adios_file="data/shinnecock_inlet", spherical=is_spherical,
        solution_var='h', friction_law='quadratic', wd=wd, wd_alpha=args.alpha,
        #forcing=GriddedForcing("data/shinnecock_forcing.hdf5"),                 
        dt=dt, bathy_adjustment=bath_adjust, nt=nt,dramp=dramp)
    p_degree = 1
    rel_toleran=1e-5
    abs_toleran=1e-6
    max_iter=10
    relax_param = 1.0
    #time series output
    
    #expected in Lon/Lat
    # the boundary node maybe
    #stations = np.array([[-72.9240934829 ,  40.7116348764, 0.0]])
    # the node in middle of channel
    stations = np.array([[-72.476519,40.840969,0.0]])
    #not sure where this node is
    #stations = np.array([[-72.0576782709, 40.9902316949, 0.0]])
    #transform to projected
    
    stations = np.deg2rad(stations) 
    # use the equator as a reference point
    lat0=35
    stations[:, 0] *= R*np.cos(np.deg2rad(lat0))
    stations[:, 1] *= R
    print(stations)

    #create solver object
    theta=1
    #first number is continuity, second is momentum
    p_degree = [1,1]
    name=args.solver.upper()
    #supg
    if name=='SUPG':
        solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
    elif name=='DG':
        #dg DGImplicit
        solver = Solvers.DGImplicit(prob,theta, p_degree=p_degree)
    elif name=='DGCG':
        #Dg continuity CG momentum
        solver = Solvers.DGCGImplicit(prob,theta, p_degree=p_degree)
    params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "bjacobi"}#,"pc_factor_mat_solver_type":"mumps"}
    
    #time and print
    start_time = time.time()
    plot_name = 'shinnecock' if not wd else f'shinnecock-wd-{args.alpha}'
    solver.time_loop(solver_parameters=params,
                        plot_every=1,
                        plot_name=plot_name,
                        stations = stations
                    )
    print("---------Simulation finished with run time %s seconds -------------" % (time.time() - start_time) )
    #verifying boundary node 75
    nbfr = 5
    eta_input = np.zeros(nt+1)
    #since boundary is one dt ahead
    t = np.linspace(0,t_f,nt+1)

    #attributes of tidal constituents at top right corner
    nodal_factors = np.array([1.021,1.021,1.000,0.947,0.913])
    rad_freq = np.array([0.000140518902509,0.000137879699487,0.000145444104333,0.000072921158358, 0.000067597744151])
    equil_args = np.array([98.846,285.394,360.000, 32.493, 70.357])
    amplitudes = np.array([0.55837173,0.13162621,0.08212716,0.07279079,0.05375799 ])
    phases = np.array([345.700,331.900,14.428,169.833,184.047])


    amplitudes = np.array([0.44836049, 0.11585067,  0.07134235, 0.06428241 ,0.05777383])
    phases = np.array([343.380, 335.853 ,18.367, 180.254, 189.278])

    equil_args = np.deg2rad(equil_args)
    phases = np.deg2rad(phases)


    for i in range(nbfr):
        #add the dt in there since boundary is one dt ahead of t?
        eta_input+=nodal_factors[i]*amplitudes[i]*np.cos(rad_freq[i]*(t) + equil_args[i] - phases[i])


    if rank ==0:
        f_extension = "spherical.png"
        #print('output station vals:',solver.vals)
        #mannually subtract
        h_b_offset = 2.1845238209
        #solver.vals[:,0,0]=solver.vals[:,0,0]-h_b_offset-bath_adjust

        #save shinecock results
        
        np.savetxt(f"{name}_p1_shinnecock_h.csv", solver.vals[:,:,0], delimiter=",")
        np.savetxt(f"{name}_p1_shinnecock_xvel.csv", solver.vals[:,:,1], delimiter=",")
        np.savetxt(f"{name}_p1_shinnecock_yvel.csv", solver.vals[:,:,2], delimiter=",")

        label=name+" solver"
        plt.plot(np.linspace(0,t_f,nt+1), solver.vals[:nt+1,0,0], "k", linewidth=2, label=label)
        plt.plot(np.linspace(0,t_f,nt+1), eta_input, "bo", linewidth=2, label="exact data")

        plt.grid(True)
        plt.xlabel("t(s)")
        plt.title('Height for Shinnecock Inlet Case in the middle of the channel.')
        plt.legend()
        plt.savefig("shinnecock_height_"+f_extension)



        plt.close()


        plt.plot(np.linspace(0,t_f,nt+1), solver.vals[:nt+1,0,1], "k", linewidth=2, label=label)

        plt.grid(True)
        plt.xlabel("t(s)")
        plt.title('X Vel for Shinnecock Inlet Case')
        plt.legend()
        plt.savefig("shinnecock_xvel_"+f_extension)

        plt.close()

        plt.plot(np.linspace(0,t_f,nt+1), solver.vals[:nt+1,0,2], "k", linewidth=2, label=label)

        plt.grid(True)
        plt.xlabel("t(s)")
        plt.title('Y Vel for Shinnecock Inlet Case')
        plt.legend()
        plt.savefig("shinnecock_yvel_"+f_extension)


