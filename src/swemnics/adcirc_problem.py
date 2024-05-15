from swemnics import problems as Problems
from mpi4py import MPI
from dataclasses import dataclass
import adios4dolfinx
from dolfinx import fem as fe, mesh, cpp
import pyvista
import dolfinx.plot
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from swemnics.boundarycondition import BoundaryCondition
import json
from scipy.spatial import KDTree
from swemnics.constants import R, earth_elasticity, g
import os

class ADCIRCTidalPotential:
    """Handles computation of body tidal forcing
    """

    def __init__(self, potential, mesh, lat, lon):
        """Initialize the class with tidal constituent information.
        """

        params = ['amplitude', 'frequency', 'nodal_factor', 'reduction_factor', 'equib_arg']
        self.constituents = []
        for p in params:
            setattr(self, p, np.zeros(len(potential)))

        i = 0
        for constituent, data in potential.items():
            self.constituents.append(constituent)
            for p in params:
                getattr(self, p)[i] = data[p]
            i += 1

        self.equib_arg = np.deg2rad(self.equib_arg)
        # create a variable parameter t
        self.t = fe.Constant(mesh, ScalarType(0))
        # construct the expression
        expr = None
        L_1 = ufl.sin(2*lon)
        L_2 = pow(ufl.cos(lat),2)
        for i, constituent in enumerate(self.constituents):
            if constituent.endswith("1"):
                L = L_1
                j = 1
            elif constituent.endswith("2"):
                L = L_2
                j = 2
            else:
                raise ValueError(f"Cannot determine tidal constituent family for '{constituent}'")

            term = (
                    earth_elasticity * self.nodal_factor[i] * self.amplitude[i] *
                    L * ufl.cos(self.frequency[i] * self.t + j * lat + fe.Constant(mesh, ScalarType(self.equib_arg[i])))
            )
            if not i: expr = term
            else: expr += term

        self.potential = expr


    def evaluate(self, t):
        """Update the tidal potential
        """
        self.t.value = t

class ADCIRCBoundaries:
    """A class to handle boundary conditions from ADCIRC
    """
    BOUNDARY_TYPES = ["open", "wall"]
    TOL = 1e-6

    def __init__(self, info,dramp):
        
        self.trees = {}
        self.bound_num = {}
        for boundary_type in self.BOUNDARY_TYPES:
            coords_list = []
            bound_num_list = []
            for i, boundary in enumerate(info[boundary_type]):
                coords_list.append(np.array(boundary["coords"]))
                bound_num_list.append(np.full(len(boundary["coords"]), i))

            self.trees[boundary_type] = KDTree(np.vstack(coords_list))
            self.bound_num[boundary_type] = np.hstack(bound_num_list)

        forcing = info["tidal_forcing"]
        num_constituents = len(forcing)
        num_nodes = len(self.bound_num["open"])
        self.amplitude = np.zeros((num_constituents, num_nodes))
        self.phase = np.zeros_like(self.amplitude)
        self.nodal_factor = np.zeros(num_constituents)
        self.equib_arg = np.zeros_like(self.nodal_factor)
        self.frequency = np.zeros_like(self.nodal_factor)
        i = 0
        for tide, constituent in forcing.items():
            self.amplitude[i] = np.array(constituent["amplitude"])
            self.phase[i] = np.array(constituent["phase"]) * (np.pi / 180)
            self.nodal_factor[i] = constituent["nodal_factor"]
            self.frequency[i] = constituent["frequency"]
            self.equib_arg[i] = constituent["equib_arg"] * (np.pi / 180)
            i += 1

        self.dramp = dramp
    def get_facet_tag(self, V):
        """Get the facet tags
        """

        # first step, try to do things through facets
        tdim = V.mesh.topology.dim
        fdim = tdim-1
        
        boundary_facets = mesh.locate_entities_boundary(V.mesh,fdim,lambda x:np.full(x.shape[1],True,dtype=bool))
        # just in case
        boundary_facets = np.sort(boundary_facets)
        # quick way to locate boundary cells from facets
        # we need the cells in order to evaluate a function. . .
        #facet_geom_entities = cpp.mesh.entities_to_geometry(V.mesh._cpp_object, tdim-1, boundary_facets, False)
        #Mark change for conda
        try:
            facet_geom_entities = cpp.mesh.entities_to_geometry(V.mesh, tdim-1, boundary_facets, False)
        except TypeError:
            facet_geom_entities = cpp.mesh.entities_to_geometry(V.mesh._cpp_object, tdim-1, boundary_facets, False)
            
        facet_coords = V.mesh.geometry.x[facet_geom_entities]

        # lookup
        facet_marks = np.full_like(boundary_facets, 0)
        for marker, boundary_type in enumerate(self.BOUNDARY_TYPES, start=1):
            dists, inds = self.trees[boundary_type].query(facet_coords[..., :2])
            dists, inds = dists.reshape((-1, 2)), inds.reshape((-1,2))
            bound_num = self.bound_num[boundary_type]
            good_mask = (
                np.all(dists < self.TOL, axis=-1) &
                #(np.abs(inds[:, 0]-inds[:,1]) == 1) &
                (bound_num[inds[:, 0]] == bound_num[inds[:, 1]])
            )
            facet_marks[good_mask] = marker
            
        return mesh.meshtags(V.mesh, fdim, boundary_facets, facet_marks)

    def locate_dofs_geometrical(self, dof_coords, boundary_type="open"):
        dof_coords = dof_coords[:, :2]
        dists, inds = self.trees["open"].query(dof_coords)
        dists, inds = dists.flatten(), inds.flatten()
        # we have the closest node for each dof coordinate
        # this algorithm assumes that if a point is on an edge, the closest
        # boundary node will be one of the two nodes on that edge
        # This is a reasonable assumption for open boundaries
        bound_coords = self.trees["open"].data
        bound_num = self.bound_num["open"]
        other_inds1 = np.maximum(inds-1, 0)
        other_inds2 = np.minimum(inds+1, len(bound_coords)-1)
        bound_mask = dists < self.TOL
        # interp_weights
        interp_weights = np.zeros((len(dof_coords), 2))
        interp_inds = np.zeros((len(dof_coords), 2), dtype=int)
        interp_inds[:, 0] = inds
        interp_weights[bound_mask, 0] = 1

        for other_inds in [other_inds1, other_inds2]:
            good_mask = (
                (inds != other_inds) & 
                (bound_num[inds] == bound_num[other_inds])
            )
            # determine distance to line
            edge = bound_coords[other_inds] - bound_coords[inds]
            offset = dof_coords - bound_coords[inds]
            edge, offset = edge[good_mask], offset[good_mask]
            inner = (edge * offset).sum(axis=-1)
            edge_norm2 = (edge ** 2).sum(axis=-1)
            offset_norm2 = (offset ** 2).sum(axis=-1)
            # first_check - we're close to the line
            # second check - we're on the edge
            mask = np.abs(inner / edge_norm2**.5) > (1-self.TOL) * offset_norm2 **.5
            mask = mask & (inner >= 0) & (inner <= edge_norm2)
            good_inds = np.where(good_mask)[0][mask]
            interp_inds[good_inds, 1] = other_inds[good_inds]
            interp_weights[good_inds,1] = (inner/edge_norm2)[mask]
            interp_weights[good_inds,0] = 1 - interp_weights[good_inds,1]
            bound_mask[good_mask] |= mask

        # put this all together and extract dofs
        self.interp_inds = interp_inds[bound_mask]
        self.interp_weights = interp_weights[bound_mask]
        self.dofs = np.where(bound_mask)[0]
        return bound_mask

    def evaluate_tidal_boundary(self, t):
        """Evaluate the tidal boundary condition
        """
        
        if not hasattr(self, 'interp_inds'):
            raise ValueError("Need to call locate_dofs_geometrical first")
        inds1, inds2 = self.interp_inds[:,0], self.interp_inds[:, 1]    
        

        tides1 = self.nodal_factor[None, :] @ (self.amplitude[:, inds1] * np.cos(
            self.frequency[:, None] * t - self.phase[:, inds1] + self.equib_arg[:, None]
        ))
        tides2 = self.nodal_factor[None, :] @(self.amplitude[:, inds2] * np.cos(
            self.frequency[:, None] * t - self.phase[:, inds2] + self.equib_arg[:, None]
        ))
        #apply tides with a ramp
        return np.tanh(2.0*t/(86400.*self.dramp))*(tides1 * self.interp_weights[:, 0] + tides2 * self.interp_weights[:, 1]).flatten()


@dataclass
class ADCIRCProblem(Problems.TidalProblem):
    
    adios_file: str = "Mesh_adios/shinnecock_inlet"
    dt: float = 1.0
    nt: int = 100
    t: float = 0
    sea_surface_height: float = 0
    bathy_adjustment: float = 0
    min_bathy: float = None
    bathy_hack_func: callable = None

    def _create_mesh(self):
        engine = "BP4"
        self.mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, self.adios_file+"_mesh.bp", engine, mesh.GhostMode.shared_facet)
        V = fe.FunctionSpace(self.mesh, ("P", 1))
        self.depth = fe.Function(V)
        adios4dolfinx.read_function(self.depth, self.adios_file+"_depth.bp", engine)
        with open(self.adios_file+"_boundary.json", "r") as fp:
            info = json.load(fp)
            self.lat0 = info.get('lat0', 0)

        self.boundaries = ADCIRCBoundaries(info,self.dramp)
        if self.spherical:
            lat = fe.Function(V)
            lat.interpolate(lambda x: x[1]/R)
            lon = fe.Function(V)
            lon.interpolate(lambda x: x[0]/(R*np.cos(np.deg2rad(self.lat0))))
            self.tidal_potential = ADCIRCTidalPotential(info["tidal_potential"], self.mesh, lat, lon)

        if os.path.exists(self.adios_file+"_mannings_n.bp"):
            self.mannings_n = fe.Function(V)
            adios4dolfinx.read_function(self.mannings_n, self.adios_file+"_mannings_n.bp", engine)

    def create_bathymetry(self, V):
        if self.min_bathy is not None:
            self.depth.x.array[np.where(self.depth.x.array < self.min_bathy)] = self.min_bathy
        
        if self.bathy_hack_func is not None:
            self.bathy_hack_func(self, self.depth)

        return self.depth + fe.Constant(self.mesh, ScalarType(self.bathy_adjustment))

    def make_h_init(self, V):
        return self.h_b + self.sea_surface_height
   
    def create_tau(self, V):
        if hasattr(self, 'mannings_n'):
            self.TAU = self.mannings_n
        else:
            super().create_tau(V)

    def init_bcs(self):
        self.facet_tag = self.boundaries.get_facet_tag(self.V)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tag)
        self._boundary_conditions = [
            BoundaryCondition(
                "Open", 1, self.u_ex.sub(0), self.V.sub(0),
                bound_func = lambda x: self.boundaries.locate_dofs_geometrical(x.T)
            ),
            BoundaryCondition("Wall", 2)
        ]
        
        self.dof_open = self._boundary_conditions[0].dofs
        self.ux_dofs_closed = np.array([])
        self.uy_dofs_closed = np.array([])
        self._dirichlet_bcs = [bc._bc for bc in self.boundary_conditions if bc.type == "Open"]

    def evaluate_tidal_boundary(self, t):
        return self.boundaries.evaluate_tidal_boundary(t) + self.sea_surface_height
        
    def plot_mesh(self):
        num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cell_entities = np.arange(num_cells, dtype=np.int32)
        args = dolfinx.plot.create_vtk_mesh(
            self.mesh, self.mesh.topology.dim, cell_entities)
        grid = pyvista.UnstructuredGrid(*args)
        grid.point_data["depth"] = self.depth.x.array
        grid.set_active_scalars("depth")

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_scalar_bar=True, show_edges=True)
        plotter.view_xy()
        plotter.show()

    def make_Source(self, u, form='well_balanced'):
        """Create the forcing terms"""
        source = super().make_Source(u, form=form)
        if self.spherical:
            tidal_body_force = ufl.as_vector((
                0,
                g*self.S * self.tidal_potential.potential.dx(0),
                g*self.tidal_potential.potential.dx(1)
            ))
            return source + tidal_body_force
        else:
            return source

    def advance_time(self):        
        self.t += self.dt
        self.update_boundary()
        if self.forcing is not None:
            self.forcing.evaluate(self.t)
        self.tidal_potential.evaluate(self.t)
