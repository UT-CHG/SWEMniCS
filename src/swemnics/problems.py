"""
A set of classes defining different test cases for the shallow water equations.

Problem classes contain most of the logic for building the flux and forcing tensors. They also specify the mesh and boundary conditions for each test scenario.
"""

from dolfinx import fem as fe
try:
  from dolfinx.fem import functionspace
except ImportError:
  from dolfinx.fem import FunctionSpace as functionspace

from dolfinx import mesh,io
from mpi4py import MPI
import numpy as np
import ufl
from ufl import (dot,div, as_tensor, as_vector, inner, dx, Measure, sqrt,conditional)
try:
  from ufl import FiniteElement, VectorElement, MixedElement
  use_basix=False
except ImportError:
  use_basix=True
  import basix
  from basix.ufl import element, mixed_element

from petsc4py.PETSc import ScalarType
from swemnics.boundarycondition import BoundaryCondition,MarkBoundary
from dataclasses import dataclass
from swemnics.constants import g, R, omega, p_water, p_air
from swemnics.forcing import GriddedForcing
import scipy
import h5py


@dataclass
class BaseProblem:
    """Steady-state problem on a unit box
    """
    h_init: float = None
    vel_init: float = None
    check_solution_def: float = None
    nx: int = 10
    ny: int = 10
    #define separatley for mixed elements
    h_0: callable = lambda x: np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)+1.0
    v_0: callable  = lambda x: np.vstack([ np.ones(x.shape[1]),np.ones(x.shape[1])])
    TAU: float = 0.0
    h_b: float = 10.0
    solution_var: str = 'eta'
    friction_law: str = 'linear'
    spherical: bool = False
    # applied only if spherical is enabled
    projected: bool = True
    # path to forcing file
    forcing: GriddedForcing = None
    lat0: float = 35
    # a ramping duration for boundary forcing
    dramp: float = 1e-10
    # wetting-and-drying flag
    wd: bool = False
    # wetting-and-drying parameter
    wd_alpha: float = 0.05

    def __post_init__(self):
        """Initialize the mesh and other variables needed for BC's
        """
        
        self._create_mesh()
        self._dirichlet_bcs = None
        self._boundary_conditions = None

    def _create_mesh(self):
        self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
   
    def log(self, *msg):
        """Log a message
        """

        if self.mesh.comm.Get_rank() == 0: print(*msg)

    def init_V(self, V):
        """Initialize the space V in which the problem will be solved.
        
        This is defined in the solver, which then calls this method on initialization.
        The first subspace of V MUST correspond to h.
        """

        self.V = V
        # initialize exact solution
        self.u_ex = fe.Function(V)

        #part of rewrite for mixed element
        self.u_ex.sub(0).interpolate(self.h_0)
        self.u_ex.sub(1).interpolate(self.v_0)
        scalar_V = V.sub(0).collapse()[0]
        if self.spherical:
            # TODO - apply nondimensionalization to spherical cases
            self.S = fe.Function(scalar_V)
            self.tan = fe.Function(scalar_V)
            self.sin = fe.Function(scalar_V)
            if self.projected:
                self.S.interpolate(lambda x: np.cos(np.deg2rad(self.lat0))/np.cos(x[1]/R))
                self.tan.interpolate(lambda x: np.tan(x[1]/R))                
                self.sin.interpolate(lambda x: np.sin(x[1]/R))                
            else:
                # raw spherical
                self.S.interpolate(lambda x: 1./np.cos(x[1]))
                self.tan.interpolate(lambda x: np.tan(x[1]))
                self.sin.interpolate(lambda x: np.sin(x[1]))


        if self.forcing is not None:
            self.forcing.set_V(scalar_V)
            self.forcing.evaluate(self.t)

    def _get_standard_vars(self, u, form='h'):
        """Return a standardized representation of the solution variables
        """

        if self.solution_var == 'h':
            h, ux, uy = u[0], u[1], u[2]
            eta = h - self.h_b
            if self.wd:
                if form =='h' or form =='flux':
                    h = h + self._wd_f(h)
            else:
                print("WD NONACTIVE")
            hux, huy = h*ux, h*uy
        elif self.solution_var == 'eta':
            eta, ux, uy = u[0], u[1], u[2]
            h = eta + self.h_b
            hux, huy = h*ux, h*uy
        elif self.solution_var == 'flux':
            h, hux, huy = u[0], u[1], u[2]
            eta = h - self.h_b
            ux, uy = hux / h, huy / h
        else:
            raise ValueError(f"Invalid solution variable '{self.solution_var}'")

        if form == 'h': return h, ux, uy
        elif form == 'eta': return eta, ux, uy
        elif form == 'flux': return h, hux, huy
        else:
            raise ValueError(f"Invalid output form '{form}'")
    
    def _wd_f(self, h):
        """Compute correction to water column height for wetting-and-drying
        """
        #include option for w/d
        #this comes from Karna papr, DG w/d
        #we modify the equations by including h_tilde as the conserved variable

        #this modifying function is straight from paper, though it could be interesting
        # to modify maybe a tanh or a bubble function
        if hasattr(self, '_wd_alpha_sq'):
            wd_alpha_sq = self._wd_alpha_sq
        else:
            wd_alpha_sq = self._wd_alpha_sq = fe.Constant(self.mesh, ScalarType(self.wd_alpha**2))
        #alpha hard coded, maybe open to front end later
        return .5 * (sqrt(h**2 + wd_alpha_sq) - h)

    def get_h_b(self, u):
        """Get the modified bathymetry
        """

        if self.wd:
            if self.solution_var in ['flux', 'h']: h = u[0]
            elif self.solution_var == 'eta': h = u[0] + self.h_b
            return self.h_b + self._wd_f(h)
        return self.h_b

    def make_Fu(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        h_b = self.get_h_b(u)
        #Mark: set nc momentum by default if wd active
        #maybe in future allow for choice of conservative
        #but doesnt seem to work very well anyway
        if self.wd:
            eta, _, _ = self._get_standard_vars(u,form='eta')
            #this is not valid for DG with jumps in bath.
            components = [
                    [h*ux,h*uy], 
                    [ux*ux+ g*eta, ux*uy],
                    [ux*uy,uy*uy+g*eta]
            ]
        else:
          #well balanced from Kubatko paper
          components = [
            [h*ux,h*uy], 
            [h*ux*ux+ 0.5*g*h*h-0.5*g*h_b*h_b, h*ux*uy],
            [h*ux*uy,h*uy*uy+0.5*g*h*h-0.5*g*h_b*h_b]
          ]    
        
        if self.spherical:
            # add spherical correction factor
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    def make_Fu_wall(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        h_b = self.get_h_b(u)
        if self.wd:
            eta, _,_ = self._get_standard_vars(u,form='eta')
            components = [
                [0,0], 
                [ g*eta, 0],
                [0,g*eta]
            ]
        else:
            #for well balanced
            components = [
                [0,0], 
                [0.5*g*h*h-0.5*g*h_b*h_b, 0],
                [0,0.5*g*h*h-0.5*g*h_b*h_b]
            ]

        if self.spherical:
            # add spherical correction factor
            #Mark messing with things
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                #just write our own
                #components = [
                #    [(self.S-1)*h*ux,0], 
                #    [ (self.S-1)*(h*ux*ux)+self.S*0.5*g*h*h, 0],
                #    [(self.S-1)*h*ux*uy,0.5*g*h*h ]
                #    ]
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    

    def get_friction(self, u):
        friction_law = self.friction_law
        h, ux, uy = self._get_standard_vars(u, form='h')
        if friction_law == 'linear':
            #hard code experiment
            cf = 0.025
            self.log("CF = ",cf)
            #linear law which is same as ADCIRC option
            return as_vector((0,
                 ux*cf,
                uy*cf))
        elif friction_law == 'quadratic':
            #experimental but 1e-16 seems to be ok
            eps = 1e-16
            vel_mag = conditional(ux*ux + uy*uy < eps, eps, pow(ux*ux + uy*uy, 0.5))
            self.TAU_const = .003 
            return as_vector(
                (0,
                vel_mag*ux*self.TAU_const,
                vel_mag*uy*self.TAU_const) )

        elif friction_law == 'mannings':
            #experimental but 1e-16 seems to be ok
            eps = 1e-8
            self.TAU_const = .02
            mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, 0, pow(ux*ux + uy*uy, 0.5))
            return as_vector(
                (0,
                g*self.TAU_const*self.TAU_const*ux*mag_v*pow(h,-1/3),
                g*self.TAU_const*self.TAU_const*uy*mag_v*pow(h,-1/3)) )
        
        elif friction_law == 'nolibf2':
            eps=1e-5
            mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, eps, pow(ux*ux + uy*uy, 0.5))
            FFACTOR = 0.0025
            HBREAK = 1.0
            FTHETA = 10.0
            FGAMMA = 1.0/3.0
            self.log("USING NOLIBF2")
            Cd = conditional(h>eps, FFACTOR* ( ( 1+  (HBREAK/h)**FTHETA  )**(FGAMMA/FTHETA) ), eps)
            #Cd = conditional(h<HBREAK, FFACTOR* ( ( 1+  (HBREAK/h)**FTHETA  )**(FGAMMA/FTHETA) ), FFACTOR )
            return as_vector(
                (0,
                Cd*ux*mag_v,
                Cd*uy*mag_v) )  
        else:
            return as_vector((0,0,0))          

    
    def make_Source(self, u,form='well_balanced'):
        h, ux, uy = self._get_standard_vars(u, form='h')
        h_b = self.get_h_b(u)
        #Mark adding nc source for wd
        #h_b = self.h_b
        if self.wd:
            if self.spherical:
                if self.projected:
                    #pretty sure never used, can remove later
                    if form != 'well_balanced':
                        g_vec = as_vector(
                            (
                                0,
                                -ux*ux.dx(0)*self.S - ux*uy.dx(1)
                                -uy*uy.dx(1) - uy*ux.dx(0)*self.S))
                    #well balanced is default
                    #check math
                    else:
                        g_vec = as_vector(
                            (
                            0,#-h * uy * self.tan / R,
                            -ux*ux.dx(0)*self.S - ux*uy.dx(1), #- 2 * ux * uy * self.tan / R - 2*uy*omega*self.sin,
                            -uy*uy.dx(1) - uy*ux.dx(0)*self.S# + 2 * ux * ux * self.tan / R + 2*ux*omega*self.sin
                            )
                        )
                else:
                    #need to check math
                    if form != 'well_balanced':
                        g_vec = as_vector(
                            (
                            -h * uy * self.tan / R,
                            -ux*ux.dx(0)*self.S/R - ux*uy.dx(1) - ux * uy * self.tan / R - 2*uy*omega*self.sin,
                            -uy*uy.dx(1) - uy*ux.dx(0)*self.S/R + ux * ux * self.tan / R + 2*ux*omega*self.sin
                            )
                        )
                    #well balanced
                    else:
                        g_vec = as_vector(
                            (
                            -h * uy * self.tan / R,
                            -ux*ux.dx(0)*self.S/R - ux*uy.dx(1) - ux * uy * self.tan / R - 2*uy*omega*self.sin,
                            -uy*uy.dx(1) - uy*ux.dx(0)*self.S/R + ux * ux * self.tan / R + 2*ux*omega*self.sin
                            )
                        )
            else:
                if form != 'well_balanced':
                    #trick for SUPG only
                    print("SUPG nonspherical\n")
                    g_vec = as_vector(
                        (
                            0,
                            -g*h_b.dx(0),
                            -g*h_b.dx(1)
                        )
                    )

                #well balanced is default
                else:
                    self.log("USING NONSPHERICAL WELLBALANCED")
                    g_vec = as_vector(
                        (
                        0,
                        -ux*ux.dx(0) - ux*uy.dx(1),
                        -uy*uy.dx(1) - uy*ux.dx(0)
                        )
                    )

        #no wd
        else:
            if self.spherical:
                if self.projected:
                    #canonical form is necessary for SUPG terms
                    if form != 'well_balanced':
                        g_vec = as_vector(
                            (
                                0,#-h * uy * self.tan / R,
                                -g*h*h_b.dx(0) * self.S ,#- 2*h * ux * uy * self.tan / R - 2*uy*h*omega*self.sin,
                                -g*h*h_b.dx(1),# + 2*h * ux * ux * self.tan / R + 2*ux*h*omega*self.sin
                            )
                        )
                    #well balanced is default
                    else:
                        g_vec = as_vector(
                            (
                                -h * uy * self.tan / R,
                                -g*(h-h_b)*h_b.dx(0) * self.S - 2*h * ux * uy * self.tan / R - 2*uy*h*omega*self.sin,
                                -g*(h-h_b)*h_b.dx(1) + 2*h * ux * ux * self.tan / R + 2*ux*h*omega*self.sin
                            )
                        )
                else:
                    if form != 'well_balanced':
                        g_vec = as_vector(
                            (
                                -h * uy * self.tan / R,
                                -g*h*h_b.dx(0) * self.S / R - h * ux * uy * self.tan / R - 2*uy*h*omega*self.sin,
                                -g*h*h_b.dx(1) / R + h * ux * ux * self.tan / R + 2*ux*h*omega*self.sin
                            )
                        )
                    #well balanced
                    else:
                        g_vec = as_vector(
                            (
                                -h * uy * self.tan / R,
                                -g*(h-h_b)*h_b.dx(0) * self.S / R - h * ux * uy * self.tan / R - 2*uy*h*omega*self.sin,
                                -g*(h-h_b)*h_b.dx(1) / R + h * ux * ux * self.tan / R + 2*ux*h*omega*self.sin
                            )
                        )
            else:
                if form != 'well_balanced':
                    g_vec = as_vector(
                        (
                            0,
                            -g*h*h_b.dx(0),
                            -g*h*h_b.dx(1)
                        )
                    )

                #well balanced is default
                else:
                    self.log("USING NONSPHERICAL WELLBALANCED")
                    g_vec = as_vector(
                        (
                            0,
                            -g*(h-h_b)*h_b.dx(0),
                            -g*(h-h_b)*h_b.dx(1)
                        )
                    )



        if self.wd:
            temp = self.get_friction(u)
            fric = as_vector((temp[0],temp[1]/h,temp[2]/h))
            source = g_vec +  fric
        else:    
            source = g_vec + self.get_friction(u) 

        if self.forcing is not None:
            windx, windy, pressure = self.forcing.windx, self.forcing.windy, self.forcing.pressure
            wind_mag = pow(windx*windx + windy*windy, 0.5)
            drag_coeff = (0.75 + 0.067 * wind_mag) * 1e-3 
            if self.wd:
                wind_forcing_terms = [
                    0,
                    -drag_coeff * (p_air / p_water) * windx * wind_mag / h,
                    -drag_coeff * (p_air / p_water) * windy * wind_mag / h,
                ]
            else:
                wind_forcing_terms = [
                    0,
                    -drag_coeff * (p_air / p_water) * windx * wind_mag,
                    -drag_coeff * (p_air / p_water) * windy * wind_mag,
                ]
            #wind_forcing_terms = [0, 30*.001 * windx*wind_mag * (p_air/p_water), 30*.001 * windy *wind_mag * (p_air/p_water)] 

            #wind_vec = as_vector(wind_forcing_terms)
            #wind_form = dot(wind_vec, wind_vec) * dx
            #print("Initial wind forcing", fe.assemble_scalar(fe.form(wind_form))**.5)
            #raise ValueError()
            if self.wd:
                pressure_forcing_terms = [
                    0,
                    pressure.dx(0) / (p_water),
                    pressure.dx(1) / (p_water)
                ]
            else:
                pressure_forcing_terms = [
                    0,
                    h * pressure.dx(0) / (p_water),
                    h * pressure.dx(1) / (p_water)
                ]
            if self.spherical:
                pressure_forcing_terms[1] *= self.S
                if not self.projected:
                    pressure_forcing_terms[1] /= R
                    pressure_forcing_terms[2] /= R

            source += as_vector(wind_forcing_terms) + as_vector(pressure_forcing_terms)
            #source += as_vector(pressure_forcing_terms)

        return source

    def make_Fu_linearized(self,u):
        '''
        routine for computing momentum flux for linearized swe
        as used for manufactured solution test cases
        '''
        h, ux, uy = self._get_standard_vars(u, form='h')
        h_b = self.get_h_b(u)
        components = [
            [h_b*ux,h_b*uy], 
            [g*h-g*h_b, 0.0],
            [0.0,g*h-g*h_b]
        ]

        if self.spherical:
            # add spherical correction factor
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    def make_Fu_top_wall_linearized(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        h_b = self.get_h_b(u)
        #this would be u dot n =0 
        components = [
            [h_b*ux,0], 
            [g*h-g*h_b, 0],
            [0,g*h-g*h_b]
        ]
        return as_tensor(components)

        
    def make_Fu_side_wall_linearized(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        h_b = self.get_h_b(u)
        #this would be u dot n =0 
        components = [
            [0,h_b*uy], 
            [g*h-g*h_b, 0],
            [0,g*h-g*h_b]
        ]
        
        if self.spherical:
            # add spherical correction factor
            #Mark messing with things
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    def make_Source_linearized(self, u,form='well_balanced'):
        h, ux, uy = self._get_standard_vars(u, form='h')
        #just a linear friction term
        cf = 0.0001
        #cf=0.000001
        print("Linear source terms!! Using friction coefficient of ",cf)
        #linear law which is same as ADCIRC option
        return as_vector((0,
                    ux*cf,
                    uy*cf))

        
    def init_bcs(self):
        """Create the boundary conditions
        """
        
        def open_boundary(x):
        	return np.isclose(x[0],0) | np.isclose(x[1],0)

        def closed_boundary(x):
        	return np.isclose(x[0],1) | np.isclose(x[1],1)

        # dealing with a vector u formulation, so adapt accordingly
        dofs_open = fe.locate_dofs_geometrical((self.V.sub(0), self.V.sub(0).collapse()[0]), open_boundary)[0]
        
        bcs = [fe.dirichletbc(self.u_ex.sub(0), dofs_open)]

        ux_dofs_closed = fe.locate_dofs_geometrical((self.V.sub(1), self.V.sub(1).collapse()[0]), closed_boundary)[0]
        uy_dofs_closed = fe.locate_dofs_geometrical((self.V.sub(2), self.V.sub(2).collapse()[0]), closed_boundary)[0]
        bcs += [fe.dirichletbc(self.u_ex.sub(1), ux_dofs_closed), fe.dirichletbc(self.u_ex.sub(2), uy_dofs_closed)]
        self._dirichlet_bcs = bcs

    def get_rhs(self):
        """Return the RHS (forcing term)
        """

        return div(self.make_Fu(self.u_ex))

    def l2_norm(self, vec):

        return (fe.assemble_scalar(fe.form(inner(vec, vec)*dx)))**.5


    def reverse_projection(self, coords):
        """Convert back to original coordinate system
        """
        if not self.projected: return coords
        else: return np.rad2deg(coords / np.array([[R*np.cos(np.deg2rad(self.lat0)), R]]))

    def apply_projection(self, coords):
        """Convert to projected coordinate system
        """
        if self.spherical:
            coords = np.deg2rad(coords)
            if self.projected:
                coords[:, 0] *= R*np.cos(np.deg2rad(self.lat0))
                coords[:, 1] *= R
        return coords


    def plot_solution(self,u_sol,filename,t=0):
        #takes a function and plots as 
        xdmf = io.XDMFFile(self.mesh.comm, filename+"/"+filename+".xdmf", "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(u_sol,t)
        xdmf.close()

    def error_infinity(self,u_h, u_ex):
        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        comm = u_h.function_space.mesh.comm
        #u_ex_V = fe.Function(u_h.function_space)
        '''
        if isinstance(u_ex, ufl.core.expr.Expr):
            u_expr = fe.Expression(u_ex, u_h.function_space.element.interpolation_points)
            u_ex_V.interpolate(u_expr)
        else:
        '''
        #u_ex_V.interpolate(u_ex)
        # Compute infinity norm, furst local to process, then gather the max
        # value over all processes
        error_max_local = np.max(np.abs(u_h.x.array - u_ex.x.array))
        error_max = comm.allreduce(error_max_local, op=MPI.MAX)
        return error_max

    @property
    def dirichlet_bcs(self):
        if self._dirichlet_bcs is None:
            self.init_bcs()

        return self._dirichlet_bcs

    @property
    def boundary_conditions(self):
        if self._boundary_conditions is None:
            self.init_bcs()

        return self._boundary_conditions

    @property
    def g(self):
        """Return the nondimensional version of g
        """

        return g


@dataclass
class TidalProblem(BaseProblem):
    """ Tidal problem on a rectangle domain, adapted from: A stable space-time FE method for the shallow water equations
       Eirik Valseth
    """
    
    dt: float = 1.0
    nt: int = 100
    TAU: float = 0.0025
    solution_var: str = 'h'
    friction_law: str = 'quadratic'
    x0: float = 0
    x1: float = 10000
    y0: float = 0
    y1: float = 2000
    mag: float = 0.15
    alpha: float = 0.00014051891708
    h_b: float = 10.0
    t: float = 0    
    
    def _create_mesh(self):
        """Initialize the mesh and other variables needed for BC's
        """
        """for now, hard coded size of mesh
        """

        self.mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[self.x0, self.y0],[self.x1, self.y1]], [self.nx, self.ny])
        self.boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.logical_not(np.isclose(x[0],self.x0 )) | np.isclose(x[1],self.y1) |  np.isclose(x[1],self.y0))]


    def create_bathymetry(self, V):
        """Create bathymetry over a given FunctionSpace
        """

        h_b = fe.Function(V.sub(0).collapse()[0])
        h_b.interpolate(lambda x: 10 + x[0]*0)
        return h_b

    def create_tau(self,V):
        """Initialise the bottom roughness
        """
        # this is needed to be robust to 
        # repeated calls to init_V
        # which happens if an adaptive mesh is used
        if not hasattr(self, '_TAU'):
            self._TAU = self.TAU
            
        if self.friction_law in ['linear', 'mannings','nolibf2','quadratic']:
            self.TAU_const = fe.Constant(self.mesh, ScalarType(self._TAU))

    def make_h_init(self, V):
        return self.h_b 

    def init_V(self, V):
        """Initialize the space V in which the problem will be solved.
        
        This is defined in the solver, which then calls this method on initialization.
        The first subspace of V MUST correspond to h.
        """
        super().init_V(V)
        self.V = V
        self.u_ex = fe.Function(V)
        self.init_bcs()        
        self.h_b = self.create_bathymetry(V)
        self.h_init = self.make_h_init(V.sub(0).collapse()[0])
        self.create_tau(V)
        self.update_boundary()


    def init_bcs(self):
        """Initialize the boundary conditions
        """

        facet_markers, facet_tag = MarkBoundary(self.mesh, self.boundaries)
        self.facet_tag = facet_tag
        #generate a measure with the marked boundaries
        #save as an attribute to the class
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
        # Define the boundary conditions and pass them to the solver
        boundary_conditions = []
        V_boundary = self.u_ex.function_space
        self.dof_open = np.array([])
        self.ux_dofs_closed = np.array([])
        self.uy_dofs_closed = np.array([])
        for marker, func in self.boundaries:
            if marker == 1:
                bc = BoundaryCondition("Open", marker, self.u_ex.sub(0),
                                      V_boundary.sub(0),
                                      bound_func=func,
                                      facet_tag=facet_tag
                     )
                self.dof_open = bc.dofs
            elif marker == 2:
                bc = BoundaryCondition("Wall", marker, self.u_ex.sub(1),
                                      V_boundary.sub(1),
                                      bound_func=func,
                                      facet_tag=facet_tag
                )
            elif marker == 3:
                bc = BoundaryCondition("Open", marker, self.u_ex.sub(1),
                                      V_boundary.sub(1),
                                      bound_func=func,
                                      facet_tag=facet_tag
                )
                self.ux_dofs_closed = bc.dofs
            elif marker == 4:
                bc = BoundaryCondition("OF", marker, self.u_ex.sub(1),
                                      V_boundary.sub(1),
                                      bound_func=func,
                                      facet_tag=facet_tag
                )

            boundary_conditions.append(bc)

        self._boundary_conditions = boundary_conditions
        self._dirichlet_bcs = []#[bc._bc for bc in self.boundary_conditions if bc.type == "Open"]

    def advance_time(self):        
        self.t += self.dt
        self.update_boundary()
        if self.forcing is not None:
            self.forcing.evaluate(self.t)

    def evaluate_tidal_boundary(self, t):
        #hard code a ramp
        return np.tanh(2.0*t/(86400.*self.dramp))*self.mag*np.cos(t*self.alpha)
        #return self.mag*np.sin(t*self.alpha)

    def update_boundary(self):
        tide = self.evaluate_tidal_boundary(self.t)

        if self.solution_var == 'eta':
            self.u_ex.sub(0).x.array[self.dof_open] = tide
        else:
            if not hasattr(self, '_hb_boundary'):
                h_ex = self.u_ex.sub(0)
                h_ex.interpolate(
                    fe.Expression(
                        self.h_b,
                        self.V.sub(0).element.interpolation_points()
                    )
                )
                self._hb_boundary = h_ex.x.array[self.dof_open]

            bc = self._hb_boundary + tide
            self.u_ex.sub(0).x.array[self.dof_open] = bc

@dataclass
class IdealizedInlet(TidalProblem):

    xdmf_file: str = None
    bypass_xdmf: bool = False
    h_b: float = 14.0

    def _create_mesh(self):

        # Try directly reading the h5 file
        # on some cases xdmf has a bug that causes it to use all the memory and crash the system. . .
        # Note this option should be used in serial ONLY
        if self.bypass_xdmf:
          gdim, shape, degree = 2, "triangle", 1
          if use_basix:
            element = basix.ufl.element("Lagrange", shape, degree, shape=(2,))
          else:
            cell = ufl.Cell(shape, geometric_dimension=gdim)
            element = ufl.VectorElement("Lagrange", cell, degree)
          fname = self.xdmf_file.replace(".xdmf", ".h5")
          with h5py.File(fname, "r") as ds:
            geom = ds["Mesh/mesh/geometry"][:]
            if geom.shape[-1] == 2:
              new_geom = np.zeros((geom.shape[0], 3))
              new_geom[:, :2] = geom
              geom = new_geom
            topology = ds["Mesh/mesh/topology"][:]
            self.mesh = mesh.create_mesh(MPI.COMM_WORLD, topology, geom, ufl.Mesh(element))
        else:
          #read in xdmf file for mesh
          encoding = io.XDMFFile.Encoding.HDF5
          with io.XDMFFile(MPI.COMM_WORLD, self.xdmf_file, "r", encoding=encoding) as file:
            self.mesh = file.read_mesh()

        self.boundaries = [(1, lambda x: np.isclose(x[1], 0)),
            (2, lambda x: np.logical_not(np.isclose(x[1],0)) | np.logical_and(np.isclose(x[1],0),np.isclose(x[0],0)) |  np.logical_and(np.isclose(x[1],0),np.isclose(x[0],50000))  )]

    def create_bathymetry(self,V):
        h_b = fe.Function(V.sub(0).collapse()[0])        
        h_b.interpolate(lambda x: 5*(x[1]>=20000) + (14 - 9/20000*x[1])*(x[1]<20000) )
        return h_b

@dataclass
class WellBalancedProblem(TidalProblem):        
    """ Problem on a square domain with an extruded square
    """
    dt: float = 1.0
    nt: int = 100
    TAU: float = 0.03
    solution_var: str = 'h'
    friction_law: str = 'quadratic'
    x0: float = 0
    x1: float = 1000
    y0: float = 0
    y1: float = 1000
    mag: float = 0.15
    alpha: float = 0.00014051891708
    h_b: float = 10.0
    t: float = 0

    def _create_mesh(self):
        """Initialize the mesh and other variables needed for BC's
        """
        """for now, hard coded size of mesh
        """
        print('nx,ny cells',self.nx,self.ny)

        self.mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[self.x0, self.y0],[self.x1, self.y1]], [self.nx, self.ny])
        #entire boundary walled
        self.boundaries = [(2, lambda x: np.logical_not(x[0]<-5)) ]

    def create_bathymetry(self,V):
        h_b = fe.Function(V.sub(0).collapse()[0])
        h_b.interpolate(lambda x: 10 - 5*(np.logical_and(np.logical_and(np.logical_and(x[1]>400, x[1]<600),x[0]>400),x[0]<600))  )
        return h_b

    def evaluate_tidal_boundary(self):
        # no tides
        return 0


#a very simple forcing class to apply rainfall
class RainForcing:
    def __init__(self, mesh, rate, final_t):
        self.rate = fe.Constant(mesh, ScalarType(rate))
        self.final_t = fe.Constant(mesh, ScalarType(final_t))
        self.t = fe.Constant(mesh, ScalarType(0))


        self.rain_source = ufl.conditional(self.t <= self.final_t, self.rate,fe.Constant(mesh, ScalarType(0)))
    def evaluate(self, t):
        """Update the tidal potential
        """
        self.t.value = t




@dataclass
class RainProblem(TidalProblem):        
    """ Problem form Namo Paper
    """
    dt: float = 600.0
    nt: int = 288
    TAU: float = 0.001
    solution_var: str = 'h'
    friction_law: str = 'quadratic'
    x0: float = 0
    x1: float = 50000
    y0: float = 0
    y1: float =  8000
    mag: float = 0.15
    alpha: float = 0.00014051891708
    h_b: float = 10.0
    t: float = 0
    rain_rate: float = 7.0556*(10**-6)
    t_final: float = 86400.0
    nx: int = 25
    ny: int = 4

    def _create_mesh(self):
        """Initialize the mesh and other variables needed for BC's
        """
        """for now, hard coded size of mesh
        """
        print('nx,ny cells',self.nx,self.ny)

        self.mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[self.x0, self.y0],[self.x1, self.y1]], [self.nx, self.ny])
        self.boundaries = [(2, lambda x: np.logical_not(x[0]<-5)) ]

        self.rain = RainForcing(self.mesh, self.rain_rate, self.t_final)

    def create_bathymetry(self,V):
        h_b = fe.Function(V.sub(0).collapse()[0])
        h_b.interpolate(lambda x: 5 - np.exp(-100*(x[0]-25000)**2) )

        return h_b

    def evaluate_tidal_boundary(self,t):
        # no tides
        return 0

    def make_Source(self, u, form='well_balanced'):
        """Create the forcing terms"""
        source = super().make_Source(u, form=form)
        
        rain_source = ufl.as_vector((
                self.rain.rain_source,
                0,
                0
            ))
        return source - rain_source
    
    def update_boundary(self):
        tide = self.evaluate_tidal_boundary(self.t)


    def advance_time(self):        
        self.t += self.dt
        self.update_boundary()
        if self.forcing is not None:
            self.forcing.evaluate(self.t)
        self.rain.evaluate(self.t)

@dataclass
class DamProblem(TidalProblem):        
    """ Problem on a square domain with an extruded square
    """
    dt: float = 0.5
    nt: int = 80
    solution_var: str = 'h'
    friction_law: str = 'none'
    x0: float = 0
    x1: float = 1000
    y0: float = 0
    y1: float = 1000
    mag: float = 0.5
    alpha: float = 0.00014051891708
    floor: float = 2.0
    h_b: float = 2.0
    t: float = 0
    
    def _create_mesh(self):
        """Initialize the mesh and other variables needed for BC's
        """
        """for now, hard coded size of mesh
        """
        print('nx,ny cells',self.nx,self.ny)
        # for nondimensionalization
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        self.mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0],[x1, y1]], [self.nx, self.ny])
        #self.boundaries = [(1, lambda x: np.isclose(x[0],x1)),
        #      (2, lambda x: np.logical_not(np.isclose(x[0],0 )) & np.logical_not(np.isclose(x[0],x1 )) & (np.isclose(x[1],y1) |  np.isclose(x[1],y0))),
        #      (3, lambda x: np.isclose(x[0],0))]

        self.boundaries = [(1, lambda x: np.isclose(x[0],self.x1)),
              (2, lambda x: np.logical_not(np.isclose(x[0],0 )) | np.logical_not(np.isclose(x[0],self.x1 ))| np.isclose(x[1],self.y1) |  np.isclose(x[1],self.y0)),
              (3, lambda x: np.isclose(x[0],0))]
        print("created mesh and boundaries")

    def create_bathymetry(self,V):
        h_b = fe.Function(V.sub(0).collapse()[0])
        h_b.interpolate(lambda x: self.floor +  0*(x[0])  )
        return h_b

    def init_V(self, V):
        super().init_V(V)
        #Discontinuous bathy
        #note, only makes sense for DG
        V_bath = functionspace(self.mesh, ("DG", 0))
        self.h_init = self.make_h_init(V_bath)
    
    def make_h_init(self, V):
        h_init =  fe.Function(V)        
        h_init.interpolate(lambda x: self.floor - (self.mag)*(x[0]>500))

        return h_init
    
    def update_boundary(self):
        if self.solution_var == 'eta':
            def interp(x):
                res = np.zeros(x.shape[1])
                return res#
            self.u_ex.sub(0).interpolate(
               interp
           )
        else:
            self.u_ex.sub(0).interpolate(lambda x: 2.0-self.mag*(x[0]>500)  )
            self.u_ex.sub(1).interpolate(
                    fe.Expression(
                        ufl.as_vector([
                            fe.Constant(self.mesh, ScalarType(0)),
                            fe.Constant(self.mesh, ScalarType(0))
                        ]),
                        self.V.sub(1).element.interpolation_points())
                )
    
    def get_analytic_solution(self,x,t):
        hl = 2.0
        hr = 1.5
        x1 = 1000.0
        x0 = x1/2.0
        g=9.81
        nx=1000
        def find_hm(hm):
            return -8*g*hr*g*hm*(np.sqrt(g*hl) - np.sqrt(g*hm))**2+ (g*hm -g*hr)**2*(g*hm + g*hr)
        hm = scipy.optimize.fsolve(find_hm, (hl+hr)/2)
        #cm is
        cm = np.sqrt(g*hm) 
        xa = x0 - t*np.sqrt(g*hl)
        xb = x0 + t*(2*np.sqrt(g*hl) -3*cm)
        xc = x0 + t*(2*cm**2*(np.sqrt(g*hl) -cm ) )/(cm**2 - g*hr)

        x = np.stack((x,x),axis=1).T

        def analytic_h(x):
            #now compute possible values
            #now compute possible values
            return np.less(x[0],xa)*hl + np.greater(x[0],xc)*hr +\
                np.logical_and(np.less_equal(xa,x[0]),np.less_equal(x[0],xb))*(4/(9*g)*(np.sqrt(g*hl) - (x[0] - x0)/(2*t))**2)\
                + np.logical_and(np.less(xb,x[0]),np.less_equal(x[0],xc))*(cm**2/g)

        def analytic_u(x):
            #now compute possible values
            return np.logical_and(np.less_equal(xa,x[0]),np.less_equal(x[0],xb))*(2.0/3.0*((x[0]-x0)/t + np.sqrt(g*hl)))\
                + np.logical_and(np.less(xb,x[0]),np.less_equal(x[0],xc))*(2*(np.sqrt(g*hl) - cm))\
                + np.less(x[0],xa)*0 + np.greater(x[0],xc)*0

        h = analytic_h(x)
        u = analytic_u(x)
        return h,u



@dataclass
class ConvergenceProblem(TidalProblem):        
    """ Convergence test case
    """
    dt: float = 1.0
    nt: int = 100
    TAU: float = 0.0001
    solution_var: str = 'h'
    friction_law: str = 'quadratic'
    x0: float = 0
    x1: float = 90000
    y0: float = 0
    y1: float = 45000
    mag: float = 0.3
    alpha: float = 0.00014051891708
    h_b: float = 3.0
    t: float = 0
    check_solution_def: float = 1.0

    
    def _create_mesh(self):
        """Initialize the mesh and other variables needed for BC's
        """
        """for now, hard coded size of mesh
        """

        self.mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[self.x0, self.y0],[self.x1, self.y1]], [self.nx, self.ny])
        self.boundaries = [(1, lambda x: np.isclose(x[0], self.x1)),
                            (2, lambda x: np.isclose(x[1],self.y1) |  np.isclose(x[1],self.y0)),
                            (4, lambda x:  np.isclose(x[0],self.x0))]
                            #(2, lambda x: np.isclose(x[0],self.x0) | np.isclose(x[1],self.y0) | np.isclose(x[1],self.y1))]

        '''
                            (2, lambda x:  np.isclose(x[1],self.y1)| np.isclose(x[1],self.y0) ),
                            (4, lambda x:  np.isclose(x[0],self.x0)   ) ]
        '''

    def create_bathymetry(self, V):
        """Create bathymetry over a given FunctionSpace
        """

        h_b = fe.Function(V.sub(0).collapse()[0])
        h_b.interpolate(lambda x: 3.0 + x[0]*0)
        return h_b
    def evaluate_tidal_boundary(self, t):
        return self.mag*np.cos(t*self.alpha)
    
    def make_h_init(self, V):
        h_init =  fe.Function(V)
        #Compute analytic solution
        #Analytic equation
        omega=0.00014051891708
        #omega=0.00014051891708/(24.0*2.0)
        tau=0.0001
        #tau=0.000001
        g=9.81#9.1845238209
        H0=3.0
        beta2=(omega**2-omega*tau*1j)/(g*H0)
        beta = np.sqrt(beta2)
        alpha_0=0.3
        xL = 90000.0
        t=0

        
        h_init.interpolate(lambda x: H0+np.real(alpha_0*np.exp(1j*omega*t)*((np.cos(beta*(x[0])))/(np.cos(beta*xL))))  )
        
        return h_init

    def make_vel_init(self, V):
        vel_init =  fe.Function(V)
        #Analytic equation
        omega=0.00014051891708
        #omega=0.00014051891708/(24.0*2.0)
        tau=0.0001
        #tau=0.000001
        g=9.81#9.1845238209
        H0=3.0
        beta2=(omega**2-omega*tau*1j)/(g*H0)
        beta = np.sqrt(beta2)
        alpha_0=0.3
        xL = 90000.0
        t=0

        vel_init.interpolate(lambda x: (np.real(-1j*omega*alpha_0/(beta*H0)*np.exp(1j*omega*t)*(np.sin(beta*(x[0])))/(np.cos(beta*(xL))) ),0*x[0]))        
        
        return vel_init
    
    def check_solution(self, u_sol,V, t):
        """Check the solution and compare to analytic solution
        """

        '''Compute analytic solution'''
        #Analytic equation
        omega=0.00014051891708
        #omega=0.00014051891708/(24.0*2.0)
        tau=0.0001
        #tau=0.000001
        g=9.81#9.1845238209
        H0=3.0
        beta2=(omega**2-omega*tau*1j)/(g*H0)
        beta = np.sqrt(beta2)
        alpha_0=0.3
        xL = 90000.0

        u_analytic =  fe.Function(V)
        u_analytic.sub(0).interpolate(lambda x: H0+np.real(alpha_0*np.exp(1j*omega*t)*((np.cos(beta*(x[0])))/(np.cos(beta*xL))))  )
        u_analytic.sub(1).interpolate(lambda x: (np.real(-1j*omega*alpha_0/(beta*H0)*np.exp(1j*omega*t)*(np.sin(beta*(x[0])))/(np.cos(beta*(xL))) ),0*x[0]))
        #pts = u_sol.split()
        #h_sol = parts[0]
        #u_sol = parts[1]

        #evaluate error L2
        e0 = u_sol.sub(0)-u_analytic.sub(0)
        l2_err=self.l2_norm(e0)
        print('L2 error for h:', l2_err)
        e1 = u_sol.sub(1)-u_analytic.sub(1)
        print('L2 error for u:', self.l2_norm(e1))

        #evaluate error Linf
        linf=self.error_infinity(u_sol.sub(0).collapse(), u_analytic.sub(0).collapse())
        print("L infinity error for h:",linf)
        linf_u=self.error_infinity(u_sol.sub(1).collapse(), u_analytic.sub(1).collapse())
        print("L infinity error for u:",linf_u)
        return l2_err

@dataclass
class SlopedBeachProblem(TidalProblem):
    #from Balzano
    #"A. Balzano, Evaluation of methods for numerical simulation of wetting and
    #drying in shallow water flow models, Coastal Engrg. 34 (1998) 83–107."
    #bathymetry_gradient: float = .1
    x0: float = 0
    x1: float = 13800
    y0: float = 0
    y1: float =  7200
    h_b_val: float = 5.0
    nx: int = 12
    ny: int = 6
    friction_law: str = 'mannings'
    mag: float = 2.0
    #period is 12 h so 2pi/43200
    alpha: float = 2.0*np.pi/(12.0*60*60)
    dramp: float = 2

    def create_bathymetry(self, V):
        """Create bathymetry over a given FunctionSpace
        """
        h_b = fe.Function(V.sub(0).collapse()[0])
        #make shore line at x1
        shoreline_x = self.x1
        self.log(f"Location of shoreline: {shoreline_x}")
        h_b.interpolate(lambda x: self.h_b_val / self.x1 * (shoreline_x - x[0]))
        return h_b
