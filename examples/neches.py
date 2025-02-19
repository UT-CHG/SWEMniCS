from swemnics import solvers as Solvers
from swemnics.adcirc_problem import ADCIRCProblem
from mpi4py import MPI
import numpy as np
from swemnics.constants import R
import time
import argparse as ap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--solver", choices=["dg", "supg"], default="supg")
    parser.add_argument("--dt", type=float, default=600)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()
    print("Running Neches")
    dt = args.dt
    t_f = 5 * 24 * 3600 - 3600
    nt = int(t_f / dt)
    is_spherical = True
    wd = args.alpha is not None
    bath_adjust = 0 if wd else 4.0
    dramp = 2.0
    prob = ADCIRCProblem(
        adios_file="data/neches_river2",
        spherical=is_spherical,
        solution_var="h",
        friction_law="mannings",
        wd=wd,
        wd_alpha=args.alpha,
        dt=dt,
        bathy_adjustment=bath_adjust,
        nt=nt,
        dramp=dramp,
    )
    p_degree = 1
    rel_toleran = 1e-5
    abs_toleran = 1e-6
    max_iter = 10
    relax_param = 1.0
    # time series output

    # expected in Lon/Lat
    # the boundary node maybe
    # stations = np.array([[-72.9240934829 ,  40.7116348764, 0.0]])
    # the node in middle of channel
    stations = np.array([[-72.476519, 40.840969, 0.0]])
    # not sure where this node is
    # stations = np.array([[-72.0576782709, 40.9902316949, 0.0]])
    # transform to projected

    stations = np.deg2rad(stations)
    # use the equator as a reference point
    lat0 = 35
    stations[:, 0] *= R * np.cos(np.deg2rad(lat0))
    stations[:, 1] *= R
    print(stations)

    # create solver object
    theta = 1
    # first number is continuity, second is momentum
    p_degree = [1, 1]
    name = args.solver.upper()
    # supg
    if name == "SUPG":
        solver = Solvers.SUPGImplicit(prob, theta, p_degree=p_degree)
    elif name == "DG":
        # dg DGImplicit
        solver = Solvers.DGImplicit(prob, theta, p_degree=p_degree)
    elif name == "DGCG":
        # Dg continuity CG momentum
        solver = Solvers.DGCGImplicit(prob, theta, p_degree=p_degree)
    params = {
        "rtol": rel_toleran,
        "atol": abs_toleran,
        "max_it": max_iter,
        "relaxation_parameter": relax_param,
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
    }  # ,"pc_factor_mat_solver_type":"mumps"}

    # time and print
    start_time = time.time()
    plot_name = "neches" if not wd else f"neches-wd-{args.alpha}"
    solver.time_loop(
        solver_parameters=params, plot_every=1, plot_name=plot_name, stations=stations
    )
    print(
        "---------Simulation finished with run time %s seconds -------------"
        % (time.time() - start_time)
    )
