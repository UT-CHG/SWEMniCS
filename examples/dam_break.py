from swemnics.problems import DamProblem
from swemnics.solvers import get_solver
from swemnics import FrictionLaw
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

    nx = ny = 100

    dt = 0.5
    # dt = 2
    t = 0
    t_f = 40
    # used in plotting
    Lx = 1000
    dam_height = 2.0

    nt = int(np.ceil(t_f / dt))
    print("Number of time steps", nt)
    # friction law either quadratic or linear
    fric_law = FrictionLaw.none
    # choose solution variable, either h or eta or flux
    sol_var = "h"

    prob = DamProblem(
        dt=dt,
        nt=nt,
        nx=nx,
        ny=ny,
        friction_law=fric_law,
        solution_var=sol_var,
        spherical=False,
    )
    p_degree = [1, 1]
    rel_toleran = 1e-5
    abs_toleran = 1e-6
    max_iter = 10
    relax_param = 1.0
    # time series output
    # time series output
    nx = 100
    stations = np.zeros((nx, 3))
    stations[:, 0] = np.linspace(0, 1000, nx)
    stations[:, 1] = 450
    # create solver object

    # cg
    theta = 1
    solver = get_solver(name)(prob, theta, p_degree=p_degree, **kwargs)

    name = name.upper()
    params = {
        "rtol": rel_toleran,
        "atol": abs_toleran,
        "max_it": max_iter,
        "relaxation_parameter": relax_param,
        "ksp_type": "gmres",
        "pc_type": "ilu",
    }  # ,"pc_factor_mat_solver_type":"mumps"}
    solver.time_loop(
        solver_parameters=params,
        stations=stations,
        plot_every=1,
        plot_name="dam_test_" + name,
    )

    # solver.solve()
    # prob.plot_solution(solver.u.sub(0),'Single_time_step')
    # print(solver.station_data.shape)
    # save array for post processing
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
    outdir = "" if outdir is None else outdir + "/"
    np.savetxt(f"{outdir}{name}_p1_stations_h.csv", solver.vals[:, :, 0], delimiter=",")
    np.savetxt(
        f"{outdir}{name}_p1_stations_xvel.csv", solver.vals[:, :, 1], delimiter=","
    )
    np.savetxt(
        f"{outdir}{name}_p1_stations_yvel.csv", solver.vals[:, :, 2], delimiter=","
    )
    if rank == 0:
        x = np.linspace(0, Lx, nx)
        plt_nums = [0, 40, nt]
        # note that station data is array with shape nt x nstattion x 3 (h,u,v)
        for a in plt_nums:
            if a > nt:
                break
            t = a * dt
            if a != 0:
                h_analytic, u_analytic = prob.get_analytic_solution(x, t)
                plt.plot(
                    x,
                    h_analytic,
                    "-",
                    linewidth=1,
                    label="h exact at " + str(int(t * 10) / 10) + "s",
                )

            plt.plot(
                np.linspace(0, 1000, nx),
                solver.vals[a, :, 0] + dam_height,
                "--",
                linewidth=2,
                label="h at " + str(int(t * 10) / 10),
            )
        # plt.plot(np.linspace(0,1000,100), solver.vals[1,:,0], linewidth=2, label="h at 100")
        # plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
        plt.grid(True)
        plt.xlabel("x(m)")
        plt.ylabel("surface elevation(m)")
        plt.title(f"Surface Elevation for {name} Scheme")
        plt.legend()
        plt.savefig(f"{outdir}dam_height_{name}_order1_dt.png")
        plt.close()

        # also plot x velocity
        for b in plt_nums:
            if b > nt:
                break
            t = b * dt

            if b != 0:
                h_analytic, u_analytic = prob.get_analytic_solution(x, t)
                plt.plot(
                    x,
                    u_analytic,
                    "-",
                    linewidth=1,
                    label="$u_x$ exact at " + str(int(t * 10) / 10),
                )

            plt.plot(
                np.linspace(0, Lx, nx),
                solver.vals[b, :, 1],
                "--",
                linewidth=2,
                label="$u_x$ at " + str(int(t * 10) / 10) + "s",
            )
        plt.grid(True)
        plt.xlabel("x(m)")
        plt.ylabel("velocity in x direction (m/s)")
        plt.title(f"Velocity for {name} Scheme")
        plt.legend()
        plt.savefig(f"{outdir}dam_velocity_{name}_order1_dt.png")
        plt.close()

    # Your statements here

    stop = timeit.default_timer()

    print("Time: ", stop - start)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("solver", choices=["cg", "supg", "dg", "dgcg"])
    args = parser.parse_args()
    run_experiment(args.solver)
