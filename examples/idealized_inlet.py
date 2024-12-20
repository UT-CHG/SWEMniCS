from swemnics.problems import IdealizedInlet
from swemnics import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import timeit

def main(cuda=False):
    """Idealized inlet example"""
    start = timeit.default_timer()


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #not used
    nx = 20
    ny = 5

    #time step in seconds
    dt = 1200#2400
    #start time
    t = 0
    #final time in seconds
    t_f = 24*60*60*4#24*11*60*60#*24*11*60*60
    nt = int(np.ceil(t_f/dt))
    print('nmber of time steps',nt)

    #friction law either quadratic or linear
    fric_law = 'quadratic'
    #choose solution variable, either h or eta or flux
    sol_var = 'h'
    #duration of ramp function in days, same as in adcirc
    dramp = 2.0

    prob = IdealizedInlet(dt=dt,nt=nt,xdmf_file='data/Ideal_Inlet/Ideal_Inlet.xdmf',friction_law=fric_law,solution_var=sol_var,dramp=dramp)
    rel_toleran=1e-9
    abs_toleran=1e-10#1e-6 reccomended for SUPG, 1e-10 for DG
    max_iter=15
    relax_param = 1.0
    p_degree=[1,1]
    plot_int = 1#np.ceil(3600/dt)
    #time series output
    stations = np.array([[25000.5,15000.5,0.0],
                            [25000.5,0.0,0.0]])
    h_b_offset = 14.0 - (9/20000)*stations[:,1]
    #stations = np.array([[25000.5,0.5,0.0]])
    #create solver object
    #cg
    theta=1
    #solver = Solvers.CGImplicit(prob,theta)
    #supg
    #solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
    #dg
    solver = Solvers.DGImplicit(prob,theta,cuda=cuda)
    #dgcg
    #solver = Solvers.DGCGImplicit(prob)
    params = {"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param}#, "ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type":"mumps"}
    solver.time_loop(solver_parameters=params,stations=stations,plot_every=plot_int,plot_name='Ideal_Inlet')



    if rank ==0:
        fout_name = "data/Ideal_inlet_adcirc_openboundary.csv"
        adcirc_dat = np.loadtxt(fout_name,delimiter=",", dtype=float)

        fout_DG = "data/DGSWEM_Ideal_inlet_adcirc.csv"
        DG_dat = np.loadtxt(fout_DG,delimiter=",", dtype=float)
        print(np.linspace(0,t_f,nt+1))
        print(solver.vals[:,0,0])
        plt.plot(np.linspace(0,t_f,nt+1), solver.vals[:nt+1,0,0], "k", linewidth=2, label="SUPG solver")
        plt.plot(adcirc_dat[:,0], adcirc_dat[:,1], "b--", linewidth=2, label="ADCIRC")
        plt.plot(DG_dat[:,0], DG_dat[:,1], "r--", linewidth = 2, label="DGSWEM")
        plt.grid(True)
        plt.xlabel("t(s)")
        plt.title('Height for Ideal Inlet Case')
        plt.legend()
        plt.savefig("inlet_height_DG.png")

        plt.close()

        plt.plot(np.linspace(0,t_f,nt+1), solver.vals[:nt+1,0,2], "k", linewidth=2, label="u at 800 m")
        plt.plot(adcirc_dat[:,0], adcirc_dat[:,3], "b--", linewidth=2, label="ADCIRC")
        #plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
        plt.grid(True)
        plt.xlabel("t(s)")
        plt.title('Velocity in y for Ideal Inlet Case')
        plt.legend()
        plt.savefig("inlet_velocity_DG.png")


        plt.close()
        mag=.15
        alpha = 0.00014051891708
        t = np.linspace(0,t_f,nt+1)
        solver.vals[:,1,0]=solver.vals[:,1,0]
        plt.plot(t, solver.vals[:nt+1,1,0], "k", linewidth=2, label="Boundary Condition")
        plt.plot(t, mag*np.cos(t*alpha), "b--",label="reference")
        plt.grid(True)
        plt.xlabel("t(s)")
        plt.legend()
        plt.savefig("inlet_bc.png")


    #Your statements here

    stop = timeit.default_timer()

    print('Time: ', stop - start)

if __name__ == "__main__":
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(cuda=args.cuda)
