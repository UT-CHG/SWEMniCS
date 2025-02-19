from swemnics.problems import ConvergenceProblem
from swemnics.solvers import get_solver
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit
import argparse as ap


def run_experiment(name):
    start = timeit.default_timer()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nx = 24  # 48#24#12#6
    ny = 12  # 24#12#6#3
    dt = 5
    t_f = 60 * 60 * 2  # * 24 * 2  # 7*24*60*60
    plot_each = 20  # 180
    # t_f = 1000
    nt = int(np.ceil(t_f / dt))
    print(f"Number of time steps {nt}")
    # friction law either quadratic or linear
    fric_law = "linear"
    # choose solution variable, either h or eta or flux
    sol_var = "h"

    # Analytic equation
    omega = 0.00014051891708
    # omega=0.00014051891708/(24.0*2.0)
    tau = 0.0001
    # tau=0.000001
    g = 9.81  # 9.1845238209
    H0 = 3.0
    beta2 = (omega**2 - omega * tau * 1j) / (g * H0)
    beta = np.sqrt(beta2)
    alpha_0 = 0.3
    xL = 90000.0

    prob = ConvergenceProblem(
        dt=dt,
        nt=nt,
        nx=nx,
        ny=ny,
        friction_law=fric_law,
        solution_var=sol_var,
        spherical=False,
    )
    p_degree = [1, 1]
    rel_toleran = 1e-11
    abs_toleran = 4e-12
    max_iter = 10
    relax_param = 1.0
    # time series output
    # time series output
    # stations = np.array([[800.5,1000.5,0.0]])
    stations = np.array([[25000.0, 22000.5, 0.0]])
    # stations = np.array([[90000.0,1000.5,0.0]])
    # compute exact solution at that point
    x_pt = stations[0][0]
    t = np.linspace(0, t_f, nt + 1)
    zeta_exact = np.real(
        alpha_0
        * np.exp(1j * omega * t)
        * ((np.cos(beta * (x_pt))) / (np.cos(beta * xL)))
    )
    vel_exact = np.real(
        -1j
        * omega
        * alpha_0
        / (beta * H0)
        * np.exp(1j * omega * t)
        * (np.sin(beta * (x_pt)))
        / (np.cos(beta * (xL)))
    )

    # overwrite
    t = np.linspace(0, t_f / (60 * 60 * 24), nt + 1)

    # create solver object

    # cg

    theta = 1
    solver = get_solver(name)(prob, theta, p_degree=p_degree, swe_type="linear")

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
        plot_every=plot_each,
        plot_name="Kubatko_test_" + name,
    )
    # print("ZETA EXACT",zeta_exact)
    # print("SOLVER VALS",solver.vals[:,:,0].flatten())
    # print("REAL FACTOR",(np.cos(beta*(x_pt)))/(np.cos(beta*xL)))
    # solver.solve()
    # prob.plot_solution(solver.u.sub(0),'Single_time_step')
    # print(solver.station_data.shape)
    # save array for post processing
    np.savetxt(f"{name}_p1_stations_eta.csv", solver.vals[:, :, 0], delimiter=",")
    np.savetxt(f"{name}_p1_stations_xvel.csv", solver.vals[:, :, 1], delimiter=",")
    np.savetxt(f"{name}_p1_stations_yvel.csv", solver.vals[:, :, 2], delimiter=",")
    if rank == 0:
        # plt_nums = [0, 1, 2, 8]
        # note that station data is array with shape nt x nstattion x 3 (h,u,v)

        plt.plot(t, zeta_exact, "k", label="Exact solution")
        plt.plot(
            t,
            solver.vals[:, :, 0].flatten(),
            "--",
            linewidth=2,
            label="h at " + str(x_pt),
        )
        # plt.plot(np.linspace(0,1000,100), solver.vals[1,:,0], linewidth=2, label="h at 100")
        # plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
        plt.grid(True)
        plt.xlabel("x(m)")
        plt.ylabel("surface elevation(m)")
        plt.title(f"Surface Elevation for {name} Scheme")
        plt.legend()
        plt.savefig(f"Tidal_height_{name}_order1_dt.png")
        plt.close()
        plt.figure()
        plt.plot(t, vel_exact, "k", label="Exact solution")
        plt.plot(
            np.linspace(0, t_f / (60 * 60 * 24), nt + 1),
            solver.vals[:, :, 1].flatten(),
            "--",
            linewidth=2,
            label=f"v at x={x_pt:.2e}",
        )
        # plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth = 2, label="Load")
        plt.grid(True)
        plt.xlabel("t(day)")
        plt.title(f"Tidal Velocity for {name} Scheme")
        plt.legend()
        plt.savefig(f"Tidal_velocity_{name}.png")

    # Your statements here

    stop = timeit.default_timer()

    print("Time: ", stop - start)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("solver", choices=["cg", "supg", "dg", "dgcg"])
    args = parser.parse_args()
    run_experiment(args.solver)
