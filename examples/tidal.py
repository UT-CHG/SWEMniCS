# # Simple tidal problem
#
#
#


import argparse
from swemnics.problems import TidalProblem
from swemnics.solvers import get_solver
from swemnics import FrictionLaw
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

parser = argparse.ArgumentParser(
    description="Simple tidal problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--nx", dest="nx", type=int, default=20, help="number of elements in x direction"
)
parser.add_argument(
    "--ny", dest="ny", type=int, default=5, help="number of elements in y direction"
)
parser.add_argument(
    "--dt", type=float, dest="dt", default=3600, help="time step in seconds"
)
parser.add_argument(
    "--t_f",
    type=float,
    dest="t_f",
    default=7 * 24 * 60 * 60,
    help="final time in seconds",
)
parser.add_argument(
    "--friction",
    dest="friction_law",
    type=FrictionLaw,
    choices=[FrictionLaw.linear, FrictionLaw.quadratic, FrictionLaw.mannings],
    help="Choice of friction law",
    default=FrictionLaw.mannings,
)
parser.add_argument(
    "--solver",
    dest="solver",
    type=str,
    default="DGNC",
    help="solver type",
    choices=["CG", "SUPG", "DG", "DGCG", "DGNC"],
)
parser.add_argument(
    "--theta",
    dest="theta",
    type=float,
    default=1.0,
    choices=[0, 0.5, 1],
    help="Time-stepping scheme: 0: Implicit Euler, 0.5: BDF2, 1: Crank-Nicholson",
)

args = parser.parse_args()


start = time.perf_counter()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


t = 0
t_f = 7 * 24 * 60 * 60  # 24*7
nt = int(np.ceil(t_f / args.dt))
print(f"Number of time steps: {nt}")

# choose solution variable, either h or eta or flux
sol_var = "h"


prob = TidalProblem(
    nx=args.nx,
    ny=args.ny,
    dt=args.dt,
    nt=nt,
    friction_law=args.friction_law,
    solution_var=sol_var,
)
p_degree = [1, 1]
rel_toleran = 1e-5
abs_toleran = 1e-6
max_iter = 10
relax_param = 1.0
# time series output
stations = np.array([[800.5, 1000.5, 0.0]])

# create solver object
solver = get_solver(args.solver)(prob, args.theta, p_degree=p_degree)

params = {
    "rtol": rel_toleran,
    "atol": abs_toleran,
    "max_it": max_iter,
    "relaxation_parameter": relax_param,
    "ksp_type": "gmres",
    "pc_type": "ilu",
    "ksp_error_if_not_converged": True,
}
if comm.size > 1:
    params["ksp_type"] = "preonly"
    params["pc_type"] = "lu"
    params["pc_factor_mat_solver_type"] = "mumps"

solver.time_loop(
    solver_parameters=params, stations=stations, plot_every=1, plot_name="SUPG_tide"
)

if rank == 0:
    # note that station data is array with shape nt x nstattion x 3 (h,u,v)
    plt.plot(
        np.linspace(0, t_f / (60 * 60 * 24), nt + 1),
        solver.vals[: nt + 1, 0, 0],
        "k",
        linewidth=2,
        label="h at 800 m",
    )
    # plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
    plt.grid(True)
    plt.xlabel("t(day)")
    plt.title(f"Tidal Height for {args.solver} Scheme")
    plt.savefig(f"tidal_height_{args.solver}.png")

    plt.close()

    plt.plot(
        np.linspace(0, t_f / (60 * 60 * 24), nt + 1),
        solver.vals[: nt + 1, 0, 1],
        "k",
        linewidth=2,
        label="u at 800 m",
    )
    # plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
    plt.grid(True)
    plt.xlabel("t(day)")
    plt.title(f"Tidal Velocity for {args.solver} Scheme")
    plt.savefig(f"tidal_velocity_{args.solver}.png")

# Your statements here
stop = time.perf_counter()
print(f"Time: {stop - start:0.5e}")
