
"""Solver classes for steady-state and time-dependent problems.

Each Solver requires a Problem class to initialize, and Solvers inherit from one another.
New numerical methods can be implemented by inheriting from the classes in this file.
"""

from dolfinx import fem as fe, nls, log,geometry,io,cpp
try:
  from dolfinx.fem import functionspace
except ImportError:
  from dolfinx.fem import FunctionSpace as functionspace

from ufl import (
    TestFunction, TrialFunction, FacetNormal, as_matrix,
    as_vector, as_tensor, dot, inner, grad, dx, ds, dS,
    jump, avg,sqrt,conditional,gt,div,nabla_div,tr,diag,sign,elem_mult,
    TestFunctions,cell_avg
)
try:
  from ufl import FiniteElement, VectorElement, MixedElement
  use_basix=False
except ImportError:
  use_basix=True
  from basix.ufl import element, mixed_element

from ufl.operators import cell_avg
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from swemnics.newton import CustomNewtonProblem, NewtonSolver
from swemnics.constants import g, R
try:
  import pyvista
  import dolfinx.plot
  have_pyvista = True
except ImportError:
  have_pyvista = False

from petsc4py.PETSc import ScalarType


class BaseSolver:
    """Defines a base solver class that solves the steady-state shallow-water equations.
    """

    def __init__(self,problem,theta=.5,p_degree=[1,1],p_type="CG",swe_type="full"):
        r"""Iniitalize the solver.
        
        Args: 
          problem: Problem class defining the mesh and boundary conditions.
          theta: Time stepping scheme parameter. The temporal derivative is approximated as
        .. math:: \frac{\partial Q}{\partial t} = \Delta t \theta (\frac{3}{2}Q_n - 2Q_{n-1} + \frac{1}{2}Q_{n-2}) + \Delta t(1-\theta)(Q_n-Q_{n-1}). 
        
            Consequently, the scheme is Implicit Euler if theta is 0, Crank-Nicolson if theta is 1, and BDF2 if theta is .5.
          p_degree: A tuple with two integers. The first indicates the polynomial degree for the mass variable, and the second the degree for the momentum variable.
          p_type: Type of element to use - either 'CG' for continuous Galerkin or 'DG' for discontinuous Galerkin.
          This is usually set by a subclass and not the user.
          swe_type: Form of the shallow water equations to solve. Either 'full' for the full nonlinear equations (default) or 'linear' for the linearized equations. In general, 'linear' should only be used in very specific circumstances, such as verifying convergence rates to an analytic solution. 
        """
        
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        self.problem = problem
        self.theta = theta
        self.p_degree = p_degree
        self.p_type = p_type
        self.names = ["eta","u","v"]
        #extra optional parameter added for linearized
        self.swe_type=swe_type
        self.log("SWE TYPE",self.swe_type)
        if self.wd:
            self.log("Wetting drying activated \n")
        else:
            self.log("Wetting drying NOT activated \n")

        self.init_fields()
        self.init_weak_form()

    @property
    def TAU(self): return self.problem.TAU

    @property
    def domain(self): return self.problem.mesh

    @property
    def wd(self): return self.problem.wd

    def init_fields(self):
        """Initialize the relevant elements, functions, and function spaces. 
        """
        
        # We generalize the code by now including mixed elements
        if not use_basix:
            el_h   = FiniteElement(self.p_type, self.domain.ufl_cell(), degree=self.p_degree[0])
            el_vel = VectorElement(self.p_type, self.domain.ufl_cell(), degree=self.p_degree[1], dim = 2)
            self.V = functionspace(self.domain, MixedElement([el_h, el_vel]))
        else:
            el_h   = element(self.p_type, self.domain.basix_cell(), degree=self.p_degree[0])
            el_vel = element(self.p_type, self.domain.basix_cell(), degree=self.p_degree[1], shape = (2,))
            self.V = functionspace(self.domain, mixed_element([el_h, el_vel]))

        #for plotting
        self.V_vel = self.V.sub(1).collapse()[0]
        self.V_scalar = self.V.sub(0).collapse()[0]
        print('V scalar',self.V_scalar)

        #split these up

        self.u = fe.Function(self.V)
        self.hel, self.vel_sol = self.u.split()

        self.p1, self.p2 = TestFunctions(self.V)

        #try this to minimize rewrite but may want to change in future
        self.p = as_vector((self.p1,self.p2[0],self.p2[1]))


    def plot_func(self, func, name="eta"):
        """Plot a function using pyvista."""
        if not have_pyvista:
           raise ValueError("pyvista not installed!")
        num_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        cell_entities = np.arange(num_cells, dtype=np.int32)
        args = dolfinx.plot.create_vtk_mesh(
            self.domain, self.domain.topology.dim, cell_entities)
        grid = pyvista.UnstructuredGrid(*args)
        #Mark change for conda
        cell_geom_entities = cpp.mesh.entities_to_geometry(self.domain, 2, cell_entities, False)
        point_cells = np.full(len(args[-1]), 0)
        for i, p in enumerate(cell_geom_entities):
            point_cells[p] = i
        data = func.eval(self.domain.geometry.x, point_cells)
        print(data.min(), np.argmin(data), )
        grid.point_data[name] = data
        grid.set_active_scalars(name)
        bad_point = self.domain.geometry.x[np.argmin(data)]
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_scalar_bar=True, show_edges=True)
        plotter.add_points(bad_point[None,:], point_size=10.0)
        plotter.view_xy()
        plotter.set_focus(bad_point)
        self.log(bad_point)
        plotter.show()

    @property
    def V(self):
        return self._V
 
    @V.setter
    def V(self, new_v):
        self._V = new_v
        self.problem.init_V(new_v)

    def init_weak_form(self):
        """Initialize the weak form.
        """

        if self.swe_type=="full":
            self.Fu = Fu = self.problem.make_Fu(self.u)
        elif self.swe_type=="linear":
            self.Fu = Fu = self.problem.make_Fu_linearized(self.u)
        n = FacetNormal(self.domain)
        self.F = -inner(Fu,grad(self.p))*dx + dot(dot(Fu, n), self.p) * ds - dot(self.problem.get_rhs(), self.p) * dx

    def solve(self,
        u_init = lambda x: np.ones(x.shape),
        solver_parameters={}
        ):

        """Solve the steady-state equation and save the result in u_sol.
        """

        # set initial guess
        self.u.interpolate(u_init)
        
        
        prob = fe.petsc.NonlinearProblem(self.F, self.u, bcs=self.problem.dirichlet_bcs)

        # the problem appears to be that the residual is humongous. . .
        res = fe.form(self.F)
        test_res = fe.petsc.create_vector(res)
        fe.petsc.assemble_vector(test_res, res)

        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, prob)
        for k, v in solver_parameters.items():
            setattr(solver, k, v)
        solver.report=True
        solver.convergence_criterion = "incremental"
        solver.error_on_nonconvergence = False
        log.set_log_level(log.LogLevel.INFO)
        solver.solve(self.u)
        
        return self.u

    def log(self, *msg):
        if self.mpi_rank == 0:
            print(*msg)

class DGSolver(BaseSolver):
    """DG steady-state solver.
    """

    def init_fields(self):
        """Initialize the variables
        """
        self.p_type = "DG"
        el = VectorElement("DG", self.domain.ufl_cell(), self.p_degree, dim=3)
        self.V_scalar = functionspace(self.domain, (self.p_type, self.p_degree))
        self.V = functionspace(self.domain, el)
        self.u = fe.Function(self.V)
        self.p = TestFunction(self.V)

    def init_weak_form(self):
        """Initialize the weak form
        """  
        super().init_weak_form()

        #add DG upwinding
        #lagrange_mult shouldn't be constant but hardcoded for this case

        C = fe.Constant(self.domain, PETSc.ScalarType(1.0))
        n = FacetNormal(self.domain)
        flux = dot(avg(self.Fu), n('+')) - 0.5*C*jump(self.u)
        self.F += inner(flux, jump(self.p))*dS


class CGImplicit(BaseSolver):
    """Base class for all time stepping solvers.
    """

    def init_fields(self):
        super().init_fields()
        self.u_n = fe.Function(self.V)
        self.u_n.name = "u_n"
        #for second order timestep need n-1
        self.u_n_old = fe.Function(self.V)
        self.u_n_old.name = "u_n_old"

    def add_bcs_to_weak_form(self):
        """Add boundary integrals to the variational form.

        This method may need to be overridden when implementing a solver with trace variables or an alternate approach to boundary conditions.
        """

        boundary_conditions = self.problem.boundary_conditions
        self.log("have boundary conditions")
        ds_exterior = self.problem.ds
        n = FacetNormal(self.domain)
        #loop throught boundary conditions to see if there is any wall conditions
        for condition in boundary_conditions:
            if condition.type == "Open":
                self.F += dot(dot(self.Fu_open, n), self.p) * ds_exterior(condition.marker)
            if condition.type == "Wall":
                self.F += dot(dot(self.Fu_wall, n), self.p)*ds_exterior(condition.marker)
            if condition.type == "OF":
                self.F += dot(dot(self.Fu_side_wall, n), self.p)*ds_exterior(condition.marker)


    def set_initial_condition(self):
        """Set the initial condition.

        The water column height is assumed to be equal to the bathymetry unless the Problem specifies a different initial condition.
        If the Problem doesn't specifiy a velocity initial condition, it is assumed to be zero.
        """
        if self.problem.solution_var =='h' or self.problem.solution_var =='flux':
            #rewrite for mixed element
            self.log("setting initial condition")
            #if the initial condition is specified set this, if not assume level starting condition
            if self.problem.h_init is None:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_b, 
                        self.V.sub(0).element.interpolation_points())
                )
            else:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_init, 
                        self.V.sub(0).element.interpolation_points())
                )
            if self.problem.vel_init is None:
                #by default assume 0 velcity everywhere    
                self.u_n.sub(1).interpolate(
                    fe.Expression(
                        as_vector([
                            fe.Constant(self.domain, ScalarType(0.0)),
                            fe.Constant(self.domain, ScalarType(0.0))
                        ]),
                        self.V.sub(1).element.interpolation_points())
                )
            else:
                #by default assume 0 velcity everywhere    
                self.u_n.sub(1).interpolate(
                    fe.Expression(
                        as_vector([
                            fe.Constant(self.domain, ScalarType(0.00)),
                            fe.Constant(self.domain, ScalarType(0.00))
                        ]),
                        self.V.sub(1).element.interpolation_points())
                )

        if self.problem.solution_var == 'eta':
            if self.problem.h_init is not None:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_init-self.problem.h_b, 
                        self.V.sub(0).element.interpolation_points())
                )

        #apply dirichlet conditions
        self.problem.update_boundary()
        if self.problem.dof_open.size != 0:
            self.u_n.x.array[self.problem.dof_open] = self.problem.u_ex.x.array[self.problem.dof_open]
        if self.problem.uy_dofs_closed.size != 0:
            self.u_n.x.array[self.problem.uy_dofs_closed] = self.problem.u_ex.x.array[self.problem.uy_dofs_closed]
        if self.problem.ux_dofs_closed.size != 0:
            self.u_n.x.array[self.problem.ux_dofs_closed] = self.problem.u_ex.x.array[self.problem.ux_dofs_closed]
    
        self.u_n_old.sub(0).x.array[:] = self.u_n.sub(0).x.array[:]
        self.u_n_old.sub(1).x.array[:] = self.u_n.sub(1).x.array[:]
    
    def init_weak_form(self):
        """Initialize the weak form.

        This method is typically overridden by any child class implementing a different numerical method.
        """
        theta = self.theta
        self.set_initial_condition()
        #create fluxes

        if self.swe_type=="full":
            self.Fu = Fu = self.problem.make_Fu(self.u)
            self.Fu_wall = Fu_wall = self.problem.make_Fu_wall(self.u)
            self.u_ex = as_vector((self.problem.u_ex[0],self.u[1],self.u[2]))
            self.Fu_open= Fu_open = self.problem.make_Fu(self.u_ex)
            self.S = self.problem.make_Source(self.u)
        elif self.swe_type=="linear":
            self.Fu = Fu = self.problem.make_Fu_linearized(self.u)
            self.Fu_wall = Fu_wall = self.problem.make_Fu_top_wall_linearized(self.u)
            self.Fu_side_wall = Fu_side_wall = self.problem.make_Fu_side_wall_linearized(self.u)
            self.S = self.problem.make_Source_linearized(self.u)
        else:
            raise Exception("Sorry, swe_type must either be linear or full, not %s" %self.swe_type)

        
        #weak form
        #specifies time stepping scheme, save it as fe.constant so it is modifiable 
        self.theta1 = theta1 = fe.Constant(self.domain, PETSc.ScalarType(theta))
        
        #start adding to residual
        self.F = -inner(Fu,grad(self.p))*dx
        self.add_bcs_to_weak_form()

        self.dt = self.problem.dt

        #add RHS to residual
        self.F += inner(self.S,self.p)*dx
        

        #add contribution from time step
        h_b = self.problem.h_b
        if self.swe_type == "full":
            self.Q = as_vector(self.problem._get_standard_vars(self.u, "flux"))
            self.Qn = as_vector(self.problem._get_standard_vars(self.u_n, "flux"))
            self.Qn_old = as_vector(self.problem._get_standard_vars(self.u_n_old, "flux"))
        elif self.swe_type == "linear":
            #if h is unkown
            self.Q = as_vector((self.u[0], self.u[1], self.u[2] ))
            self.Qn = as_vector((self.u_n[0], self.u_n[1], self.u_n[2]))
            self.Qn_old = as_vector((self.u_n_old[0], self.u_n_old[1], self.u_n_old[2] ))
        else:
            raise Exception("Sorry, swe_type must either be linear or full, not %s" %self.swe_type)
        
        #BDF2
        self.dQdt = theta1*fe.Constant(self.domain,PETSc.ScalarType(1/self.dt))*(1.5*self.Q - 2*self.Qn + 0.5*self.Qn_old) + (1-theta1)*fe.Constant(self.domain,PETSc.ScalarType(1/self.dt))*(self.Q - self.Qn)
        u = as_vector(self.problem._get_standard_vars(self.u, "h"))
        u_n = as_vector(self.problem._get_standard_vars(self.u_n, "h"))
        u_n_old = as_vector(self.problem._get_standard_vars(self.u_n_old, "h"))
        self.dQ_ncdt = theta1*fe.Constant(self.domain,PETSc.ScalarType(1/self.dt))*(1.5*u - 2*u_n + 0.5*u_n_old) + (1-theta1)*fe.Constant(self.domain,PETSc.ScalarType(1/self.dt))*(u - u_n)           

        self.F+=inner(self.dQdt,self.p)*dx

    def solve_init(self,
        #u_init = lambda x: np.ones(x.shape),
        solver_parameters={}
        ):
        """Initialize the Newton solver
        """

        #utilize the custom Newton solver class instead of the fe.petsc Nonlinear class
        Newton_obj = CustomNewtonProblem(self,solver_parameters=solver_parameters)
        return Newton_obj
    
    def solve_timestep(self,solver):
        """Solve the nonlinear problem at the current time step.

        Args:
          solver: Newton solver.
        """
        try:
            solver.solve(self.u)
        except RuntimeError:
            h_fun = self.u.sub(0).collapse()
            hvals = h_fun.x.array[:]
            min_h = hvals.min()
            print(f"Min h on process {self.mpi_rank}, {min_h}")
            bad_h = hvals < 0
            coords = h_fun.function_space.tabulate_dof_coordinates()[:, :2]
            coords = self.problem.reverse_projection(coords)
            print(f"first coords of negative h on {self.mpi_rank}", coords[bad_h][:1])
            if not self.mpi_rank: raise

 
    def update_solution(self):
        #advance boundary and
        #save new solution as previous solution
        self.u_n_old.x.array[:] = self.u_n.x.array[:]
        self.u_n.x.array[:] = self.u.x.array[:]

        #dirichlet boundary
        self.problem.advance_time()
 
        #update any possible dirichlet boundaries
        if self.problem.dof_open.size != 0:
            self.u.x.array[self.problem.dof_open] = self.problem.u_ex.x.array[self.problem.dof_open]
        if self.problem.uy_dofs_closed.size != 0:
            self.u.x.array[self.problem.uy_dofs_closed] = self.problem.u_ex.x.array[self.problem.uy_dofs_closed]
        if self.problem.ux_dofs_closed.size != 0:
            self.u.x.array[self.problem.ux_dofs_closed] = self.problem.u_ex.x.array[self.problem.ux_dofs_closed]

    def init_stations(self,points):
        #reads in recording stations and outputs points on each processor
        if len(points):
            points = np.array(points)
            # be robust to 2-d input
            if points.shape[1] < 3:
                old_points = points
                points = np.zeros((len(points), 3))
                points[:, :old_points.shape[1]] = old_points
        else:
            self.cells = []
            self.station_bathy = np.array([], dtype=float)
            return np.array([], dtype=float)
        
        domain = self.problem.mesh       
        bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions(bb_tree, points)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
        self.station_index = []
        for i, point in enumerate(points):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
                self.station_index.append(i)
        self.cells = cells
        bathy_func = fe.Function(self.V_scalar)
        bathy_func.interpolate(fe.Expression(self.problem.h_b, self.V_scalar.element.interpolation_points()))
        self.station_bathy = bathy_func.eval(points_on_proc, self.cells)
        #print("station bathy", self.station_bathy, points_on_proc)
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        return points_on_proc

    def record_stations(self,u_sol,points_on_proc):
        """saves time series at stations into a numpy array"""
        h_values = u_sol.sub(0).eval(points_on_proc, self.cells)
        if self.problem.solution_var in ['h', 'flux']: h_values -= self.station_bathy
        u_values = u_sol.sub(1).eval(points_on_proc, self.cells)
        u_values = np.hstack([h_values,u_values])
        return u_values

    def initialize_video(self,filename):
        self.xdmf = io.XDMFFile(self.problem.mesh.comm, filename+"/"+filename+".xdmf", "w")
        self.xdmf.write_mesh(self.problem.mesh)
        

        self.eta_plot = fe.Function(self.V_scalar)
        self.eta_plot.name = "eta"


        self.vel_plot = fe.Function(self.V_vel)
        self.vel_plot.name = "depth averaged velocity"
        self.bathy_plot = fe.Function(self.V_scalar)
        self.bathy_plot.name = "bathymetry"


    def finalize_video(self):
        self.xdmf.close()

    def plot_frame(self):
        """Plot a frame of the state
        """
        # dimensional scales
        #takes a function and plots as 
        #this will output a vector xyz, want to change
        self.eta_expr = fe.Expression(self.u.sub(0).collapse() - self.problem.h_b, self.V_scalar.element.interpolation_points())
        self.eta_plot.interpolate(self.eta_expr)


        #rwerite for mixed elements
        self.v_expr = fe.Expression(self.u.sub(1).collapse(), self.V_vel.element.interpolation_points())

        self.vel_plot.interpolate(self.v_expr)

        
        self.xdmf.write_function(self.eta_plot,self.problem.t)
        self.xdmf.write_function(self.vel_plot,self.problem.t)
        if not self.problem.t:
            # write bathymetry for first timestep only
            self.log("Interpolating bathymetry")
            self.bathy_plot.interpolate(fe.Expression(self.problem.h_b, self.V_scalar.element.interpolation_points()))
            self.log("Writing bathymetry")
            self.xdmf.write_function(self.bathy_plot, self.problem.t)
            self.log("Wrote bathymetry")
        

    def plot_frame_2(self):

        #takes a function and plots as 
        self.xdmf.write_function(self.u,0)


    def gather_station(self,root,local_stats,local_vals):
        comm = self.problem.mesh.comm
        rank = comm.Get_rank()
        gathered_vals = comm.gather(local_vals,root=root)
        gathered_inds = comm.gather(np.array(self.station_index,dtype=int), root=root)
        vals = []
        inds = []
        if rank == root:
            vals = np.concatenate(gathered_vals, axis=1) 
            inds = np.concatenate(gathered_inds)
        return inds,vals

    def time_loop(self,solver_parameters,stations=[],plot_every=999999,plot_name='debug_tide'):
        self.log("calling time loop")
        self.points_on_proc = local_points = self.init_stations(stations)
        self.station_data = np.zeros((self.problem.nt+1,local_points.shape[0],3))

        #set initial guess for the first time step
        self.u.x.array[:] = self.u_n.x.array[:]


        self.solver = solver = self.solve_init(solver_parameters=solver_parameters)
        #plot the initial condition
        #Mark commented, this seems to be incorrect
        #self.update_solution()

        self.log('plot every', plot_every)
        self.log('nt',self.problem.nt)

        if plot_every<=self.problem.nt:
            self.log('creating video')
            self.initialize_video(plot_name)
            self.plot_frame()
        
        if len(stations):
          self.station_data[0,:,:] = self.record_stations(self.u,local_points)

        #take first 2 steps with implicit Euler since we dont have enough steps for higher order
        self.theta1.value=0
        for a in range(min(2,self.problem.nt)):
            self.log('Time Step Number',a,'Out of',self.problem.nt)
            self.log(a/self.problem.nt*100,'% Complete')
            self.update_solution()
            #working version here
            self.solve_timestep(solver)
            
            #add data to station variable
            if len(stations):
              self.station_data[a+1,:,:] = self.record_stations(self.u,local_points)

            if a%plot_every==0 and plot_every <= self.problem.nt:
                self.plot_frame()

        #switch to high order time stepping
        self.theta1.value=self.theta
        for a in range(2, self.problem.nt):
            self.log('Time Step Number',a,'Out of',self.problem.nt)
            self.log(a/self.problem.nt*100,'% Complete')
            self.update_solution()
            #working version
            self.solve_timestep(solver)

            if len(stations):
                self.station_data[a+1,:,:] = self.record_stations(self.u,local_points)
            if a%plot_every==0:
                self.plot_frame()
        
        if plot_every <= self.problem.nt:
            self.finalize_video()

        if len(stations):
          inds, vals = self.gather_station(0,local_points,self.station_data)
        else: inds, vals = None, None

        self.vals = vals
        self.inds = inds

        #optionally evaluate and print L2 error
        if self.problem.check_solution_def is not None:
            print("Checking solution at ",self.problem.t)
            e0=self.problem.check_solution(self.u,self.V,self.problem.t)
            print("L2 error at t=",str(self.problem.t)," is ",str(e0))

        return self.u, vals


class DGImplicit(CGImplicit):
    def init_fields(self):
        """Initialize the variables
        """
        self.p_type = "DG"

        # We generalize the code by now including 2 elements
        # We generalize the code by now including mixed elements
        if use_basix:
            el_h   = element(self.p_type, self.domain.basix_cell(), degree=self.p_degree[0])
            el_vel = element(self.p_type, self.domain.basix_cell(), degree=self.p_degree[1], shape=(2,))
            self.V = functionspace(self.domain, mixed_element([el_h, el_vel]))
        else:
            el_h   = FiniteElement(self.p_type, self.domain.ufl_cell(), degree=self.p_degree[0])
            el_vel = VectorElement(self.p_type, self.domain.ufl_cell(), degree=self.p_degree[1], dim = 2)
            self.V = functionspace(self.domain, MixedElement([el_h, el_vel]))

        #for plotting
        self.V_vel = self.V.sub(1).collapse()[0]
        self.V_scalar = self.V.sub(0).collapse()[0]

        #split these up

        self.u = fe.Function(self.V)
        self.hel, self.vel_sol = self.u.split()

        self.p1, self.p2 = TestFunctions(self.V)

        #try this to minimize rewrite but may want to change in future
        self.p = as_vector((self.p1,self.p2[0],self.p2[1]))
        self.u_n = fe.Function(self.V)
        self.u_n.name = "u_n"
        #for second order timestep need n-1
        self.u_n_old = fe.Function(self.V)
        self.u_n_old.name = "u_n_old"

    def init_weak_form(self):
        """Initialize the weak form
        """  
        super().init_weak_form()
        
        #add DG upwinding
        #Not sure if this is correct or stable yet
        #simplest Lax-Friedrichs flux on F operator
        # see https://fenicsproject.discourse.group/t/lax-friedrichs-flux-for-advection-equation/4647
        
        eps=1e-16
        n = FacetNormal(self.domain)
        #Option 1: use u as the penalizing term, empirically this seems to work best
        if self.problem.solution_var == 'h' or self.problem.solution_var == 'eta':
            #maybe?
            #C = conditional(sqrt(dot(avg(self.u),avg(self.u))) > eps,sqrt(dot(avg(self.u),avg(self.u))),eps)
            
            #or this?
            #vel = as_vector((self.problem.S*self.u[1],self.problem.S*self.u[2]))
            #C = conditional(sqrt(dot(avg(vel),avg(vel))) > eps,sqrt(dot(avg(vel),avg(vel))),eps)
            #C = C+sqrt(g*avg(self.problem.S*self.u[0]))

            #attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
            #still doesnt work
            h, ux, uy = self.problem._get_standard_vars(self.u, 'h')
            vela =  as_vector((ux('+'),uy('+')))
            velb =  as_vector((ux('-'),uy('-')))
            
            vnorma = conditional(dot(vela,vela) > eps,sqrt(dot(vela,vela)),np.sqrt(eps))
            vnormb = conditional(dot(velb,velb) > eps,sqrt(dot(velb,velb)),np.sqrt(eps))

            #TODO replace conditionals with smoother transition
            C = conditional( (vnorma + sqrt(g*h('+')) ) > (vnormb + sqrt(g*h('-')) ), (vnorma + sqrt(g*h('+'))) ,  (vnormb + sqrt(g*h('-'))) ) 

            if self.problem.spherical:
                if self.problem.projected:
                    #qustion, even if we are discretizing by primitives should jump be based on flux variable or primitive?
                    #appears both work, using Q now
                    self.log('spherical projected DG!!')
                    flux = dot(avg(self.Fu), n('+')) + 0.5*C*jump(self.Q)
                else:
                    flux = dot(avg(self.Fu), n('+')) + 0.5*C*avg(self.problem.S**2/R)*jump(self.Q)
            else:
                flux = dot(avg(self.Fu), n('+')) + 0.5*C*jump(self.Q)
        #Option 2

        if self.problem.solution_var == 'flux':
        
            prop_vel = as_vector((self.u[1]/self.u[0],self.u[2]/self.u[0]))
        
            C = conditional(sqrt(dot(avg(prop_vel),avg(prop_vel))) > eps,sqrt(dot(avg(prop_vel),avg(prop_vel))),eps)

            #trying from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
            C = C+sqrt(g*avg(self.u[0]))

            n = FacetNormal(self.domain)

            flux = dot(avg(self.Fu), n('+')) + 0.5*C*jump(self.u)
        
        self.F += inner(flux, jump(self.p))*dS


class SUPGImplicit(CGImplicit):
    #try to cell avg
    def project_L2(self,f,V):
        #takes in f and projects to functionspace V
        #V = f._V
        u = TrialFunction(V)
        v = TestFunction(V) 
        a = inner(u,v)*dx
        L = inner(f,v)*dx
        problem = fe.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "cg"})
        ux = problem.solve()
        ux.vector.ghostUpdate()
        return ux


    def init_weak_form(self):
        super().init_weak_form()
        n = FacetNormal(self.domain)
        eps = 1e-8
        #time step parameter, 0.5 for crank-nicolson, 0 for implicit euler, 1 for explicit euler
        theta = self.theta
        #temporal term from residual
        dQdt = self.dQdt
        #non-conservative dQdT
        dQ_ncdt = self.dQ_ncdt

        #get element height as elementwise constant function
        tdim = self.domain.topology.dim
        num_cells1 = self.domain.topology.index_map(tdim).size_local

        try:
            h = cpp.mesh.h(self.domain, tdim, range(num_cells1))
        except TypeError:
            h = cpp.mesh.h(self.domain._cpp_object, tdim, range(num_cells1))
            
        #save as a DG function
        self.cellwise = functionspace(self.domain, ("DG", 0))
        #V1 = functionspace(domain1,("CG",1))
        height1 = fe.Function(self.cellwise)
        height1.x.array[:num_cells1] = h
        height1.vector.ghostUpdate()

  
        # tau from AdH
        alpha = 0.25
        spherical = self.problem.spherical
        #set the upwid tensor
        if self.problem.solution_var =='eta':
            #WARNING Deprecated!!!!

            # tau from AdH
            #using previous time step
            if spherical:
                factor = sqrt(self.problem.S*self.problem.S*(self.u_n[1]*self.u_n[1] + self.u_n[2]*self.u_n[2]) + g*(self.u_n[0]+self.problem.h_b))        
            else:
                factor = sqrt(self.u_n[1]*self.u_n[1] + self.u_n[2]*self.u_n[2] + g*(self.u_n[0]+self.problem.h_b))        
           
            #upwinding from non-conservative SWE seems to work best when using primitives
            T1 = as_matrix((  (self.u[1], (self.u[0]+self.problem.h_b), 0), ( g, self.u[1], 0 ), (0, 0, self.u[1])   )) 
            T2 = as_matrix((  (self.u[2],0,(self.u[0]+self.problem.h_b)), (0, self.u[2], 0 ), ( g,0,self.u[2] )   ))
            if spherical:
                if self.problem.projected:
                    T1 = T1 * self.problem.S
                else:
                    T1 = T1 * self.problem.S / R
                    T2 = T2 / R

            #need a special source for SUPG term compatible with non-conservative SWE
            S_temp = self.problem.make_Source(self.u,form='canonical')
            S_nc = as_vector((S_temp[0],S_temp[1]/(self.u[0]+self.problem.h_b), S_temp[2]/(self.u[0]+self.problem.h_b)))

        elif self.problem.solution_var == 'h':
            h, ux, uy = self.problem._get_standard_vars(self.u, 'h')
            h_n, ux_n, uy_n = self.problem._get_standard_vars(self.u_n, 'h')
            #factor from adH
            #previous time step linearizes but seems to be worse at larger time steps
            
            #cell_avg not implemented :((((
            #nonlinear but seems to help convergence
            #factor = sqrt(self.u[1]*self.u[1] + self.u[2]*self.u[2] + g*(self.u[0]))
            #eps=1e-4
            #factor = conditional(factor>eps, factor, eps)
            
            
            
            if self.swe_type=='full':


                factor = sqrt(ux_n*ux_n + uy_n*uy_n + g*(h_n))
                T1 = as_matrix((  (ux, h, 0), ( g, ux, 0 ), (0, 0, ux)   ))
                T2 = as_matrix((  (uy,0,h), (0, uy, 0 ), ( g,0,uy )   ))
            
                #need a special source for SUPG term compatible with non-conservative SWE

                
                if self.wd:
                    #need custom vector for nc
                    S_nc = self.problem.make_Source(self.u,form='canonical')
                else:
                    S_temp = self.problem.make_Source(self.u,form='canonical')
                    S_nc = as_vector((S_temp[0],S_temp[1]/h, S_temp[2]/h))
                    
            
            elif self.swe_type=='linear':
                alpha = 0.1#0.5/(2**self.p_degree[0])
                h_b = self.problem.get_h_b(self.u)
                factor = sqrt(ux_n*ux_n + uy_n*uy_n + g*(h_b))
                T1 = as_matrix((  (0, h_b, 0), ( g, 0, 0 ), (0, 0, 0)   ))
                T2 = as_matrix((  (0,0,h_b), (0, 0, 0 ), ( g,0, 0 )   ))
                #need a special source for SUPG term compatible with non-conservative SWE
                S_temp = self.problem.make_Source_linearized(self.u,form='canonical')
                S_nc = as_vector((S_temp[0],S_temp[1], S_temp[2]))
            else:
                raise Exception("Sorry, swe_type must either be linear or full, not %s" %self.swe_type)


            if spherical:
                #factor = sqrt(
                #    self.problem.S * self.problem.S * (self.u[1]*self.u[1] + self.u[2]*self.u[2])
                #    + g*(self.u[0]))
                if self.problem.projected:
                    #S_nc = as_vector((S_temp[0]/self.u[0],S_temp[1]/self.u[0], S_temp[2]/self.u[0]))
                    #factor = sqrt(self.problem.S**2*(self.u[1]*self.u[1] + self.u[2]*self.u[2]) + g*(self.u[0]))

                    factor = sqrt(self.problem.S**2*(self.u_n[1]*self.u_n[1] + self.u_n[2]*self.u_n[2]) + g*(self.u_n[0]))
                    T1 = T1*self.problem.S
                else:
                    T1 = T1 * self.problem.S / R
                    T2 = T2 / R

            #Experimental shock capturing term
            #not used currently
            
            tau_shock = alpha*height1/factor
            dQdxdPdx = elem_mult(self.Q.dx(0),self.p.dx(0))
            dQdydPdy = elem_mult(self.Q.dx(1),self.p.dx(1))
            #alternative shock capturing
            dUdxdPdx = elem_mult(self.u.dx(0),self.p.dx(0))
            dUdydPdy = elem_mult(self.u.dx(1),self.p.dx(1))
            #option 1
            #self.F += tau_shock*inner(dQ_ncdt +T1*self.u.dx(0) + T2*self.u.dx(1) + S_nc, dQdxdPdx +dQdydPdy)*dx
            #option 2
            #UNCOMMENT THIS
            #self.F += tau_shock*inner(dQ_ncdt +T1*self.problem.S*self.u.dx(0) + T2*self.u.dx(1) + S_nc, dUdxdPdx +dUdydPdy)*dx
            
            ############################################################################        
        elif self.problem.solution_var == 'flux':
            #WARNING: not used
            #factor from adH
            factor = sqrt(self.u_n[1]*self.u_n[1] + self.u_n[2]*self.u_n[2] + g*(self.u_n[0]))

            #try upwinding tensor by differentiating using conserved variables instead of solution vars, doesnt seem to be as stable
            T1 = as_matrix(( (0,1,0), ( self.u[0]*g - self.u[1]*self.u[1]/(self.u[0]*self.u[0]) , 2*self.u[1]/self.u[0] ,0 ), (-self.u[1]*self.u[2]/(self.u[0]*self.u[0]), self.u[2]/self.u[0], self.u[1]/self.u[0]) ))
            T2 = as_matrix(( (0,0,1), ( - self.u[2]*self.u[1]/(self.u[0]*self.u[0]) , self.u[2]/self.u[0] ,self.u[1]/self.u[0] ), (self.u[0]*g - self.u[2]*self.u[2]/(self.u[0]*self.u[0]), 0, 2*self.u[2]/self.u[0]) ))

        #adH Tau
        if self.problem.solution_var == 'eta' or self.problem.solution_var == 'h':
            tau_SUPG = as_vector((alpha*height1/factor, alpha*height1/factor, alpha*height1/factor)) 

        elif self.problem.solution_var == 'flux':
            #try tau from Chen
            mag_vel = sqrt(self.u_n[1]*self.u_n[1]/(self.u_n[0]*self.u_n[0]) + self.u_n[2]*self.u_n[2]/(self.u_n[0]*self.u_n[0]))
            tau_SUPG= as_vector((pow(2/self.dt + 2*mag_vel/height1,-1),pow(2/self.dt + 2*mag_vel/height1,-1),pow(2/self.dt + 2*mag_vel/height1,-1)))   



        #petrov terms for SUPG
        temp_x = as_vector(( tau_SUPG[0]*self.p[0].dx(0), tau_SUPG[1]*self.p[1].dx(0), tau_SUPG[2]*self.p[2].dx(0) ))
        temp_y = as_vector(( tau_SUPG[0]*self.p[0].dx(1), tau_SUPG[1]*self.p[1].dx(1), tau_SUPG[2]*self.p[2].dx(1) ))


        
        #Conservative residual with SUPG (doesnt seem to work as well when primitives are unkown)
        #########################################################################################    
        if self.problem.solution_var == 'flux':
            #Warning: not used
            self.F += inner(dQdt +div(self.Fu)+self.S, (T1*temp_x+T2*temp_y)  )*dx
            #attempt adding interior penalty
            #still may need work, but appears to help stability in channel case
            #####################################################################
            omega_cip = avg(1/self.u[0])
            vel = as_vector((self.u[1],self.u[2]))
            #self.F += omega_cip*avg(height1**2)*avg(dot(vel,n))*inner(jump(dot(grad(self.u),n)),jump(dot(grad(self.p),n)))*dS
            ####################################################################
        ##################################################################################


        #for primitives use non-conservative SWE as the residual
        #seems to work best
        ########################################################################
        if self.problem.solution_var == 'eta' or self.problem.solution_var == 'h':
            if spherical:
                self.F += inner(dQ_ncdt +T1*self.u.dx(0) + T2*self.u.dx(1) + S_nc, T1*temp_x+T2*temp_y)*dx 
            else:
                self.F += inner(dQ_ncdt +T1*self.u.dx(0) + T2*self.u.dx(1) + S_nc, (T1*temp_x+T2*temp_y))*dx       
            
        ######################################################################
            #attempt adding interior penalty
            #still may need work, but appears to help stability in channel case
            #####################################################################
            omega_cip = 1.0
            vel = as_vector((self.u[1],self.u[2]))
            #self.F += omega_cip*avg(height1**2)*avg(dot(vel,n))*inner(jump(dot(grad(self.u),n)),jump(dot(grad(self.p),n)))*dS
            ###################################################################

class DGCGImplicit(DGImplicit):
    #DG continuity and CG momentum with SUPG
    def init_fields(self):
        """Initialize the variables
        """
        # We generalize the code by now including 2 elements
        el_h   = FiniteElement("DG", self.domain.ufl_cell(), degree=self.p_degree[0])
        el_vel = VectorElement("CG", self.domain.ufl_cell(), degree=self.p_degree[1], dim = 2)

        self.V = functionspace(self.domain, MixedElement([el_h, el_vel]))

        #for plotting
        self.V_vel = self.V.sub(1).collapse()[0]
        self.V_scalar = self.V.sub(0).collapse()[0]
        self.log('V scalar',self.V_scalar)

        #split these up

        self.u = fe.Function(self.V)
        self.hel, self.vel_sol = self.u.split()

        self.p1, self.p2 = TestFunctions(self.V)

        #try this to minimize rewrite but may want to change in future
        self.p = as_vector((self.p1,self.p2[0],self.p2[1]))
        self.u_n = fe.Function(self.V)
        self.u_n.name = "u_n"
        #for second order timestep need n-1
        self.u_n_old = fe.Function(self.V)
        self.u_n_old.name = "u_n_old"


    def init_weak_form(self):
        #add entire SUPG weak form
        super().init_weak_form()
        


if __name__ == "__main__":
    from base_problem import BaseProblem

    n = 15
    prob = BaseProblem(n, n)
    #rel_toleran=1e-11
    #abs_toleran=1e-12
    rel_toleran=1e-6
    abs_toleran=1e-6
    max_iter=100
    #max_iter=1
    relax_param = 0.5
    solver = DGSolver(prob)
    #solver = BaseSolver(prob)
    solver.solve(solver_parameters={"rtol": rel_toleran, "atol": abs_toleran, "max_it":max_iter, "relaxation_parameter":relax_param})
    prob.check_solution(solver.u)
    prob.plot_solution(solver.u.sub(0),'DG_steady_ele')
