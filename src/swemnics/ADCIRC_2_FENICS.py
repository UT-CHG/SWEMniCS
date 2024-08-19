import ufl
#import gmsh
import numpy as np
from dolfinx import io,fem,mesh,cpp,plot
from mpi4py import MPI
import sys
import adios4dolfinx
from constants import R
#Mark change for conda compatability
#g = 9.81
#R = 6.738e+6
import argparse as ap
import json
import os

try:
  from dolfinx.fem import functionspace
except ImportError:
  from dolfinx.fem import FunctionSpace as functionspace

try:
  from ufl import FiniteElement, VectorElement
  use_basix=False
except ImportError:
  use_basix=True
  import basix
  from dolfinx.fem import coordinate_element
  from ufl.finiteelement import FiniteElement
  from ufl.pullback import identity_pullback
  from ufl.sobolevspace import H1

supported_tides = {"Q1", "O1", "P1", "K1", "N2", "M2", "S2", "K2"}

def map_coords(coords1, coords2):
    """Return a mapping that takes coords1 to coords2
    """
    inds1 = np.lexsort(coords1.T)
    inds2 = np.lexsort(coords2.T)
    mapping = np.zeros(len(coords1), dtype=np.int64)
    mapping[inds2] = inds1
    return mapping
    #print(np.allclose(coords2, coords1[mapping]))

class ADCIRCMesh:
    """Class for converting an ADCIRC mesh into a FEniCSx mesh
    """

    def __init__(self, inputsdir, projected=False, cartesian=False, lat0=35):
        """Initialize ADCIRC mesh and tidal boundary conditions from fort.14/fort.15
        """

        self.projected = projected
        self.cartesian = cartesian
        self.lat0 = lat0
        self.boundaries = {"open":[], "wall": [], "lat0": lat0}
        self._adjacency_lists = None
        self._boundary_facets = None
        self._levee_segments = []

        if os.path.exists(inputsdir+"/fort.13"):
            self._read_nodal_attributes(inputsdir+"/fort.13")
        else:
            self.nodal_attributes = {}

        self._read_mesh(inputsdir+"/fort.14")
        self._read_tides(inputsdir+"/fort.15")
        self._create_mesh()

    @property
    def adjacency_lists(self):
        """Return adjacency lists of nodes, creating if it doesn't exist
        """

        if self._adjacency_lists is None:
            self._init_graph_info()
        return self._adjacency_lists

    def _init_graph_info(self):
        """Initialize graph information
        """
        print("Constructing graph info....")
        lists = {i: set() for i in range(len(self.nodenum))}
        boundary_facets = set() 
        for i in range(len(self.nm)):
            a, b, c = self.nm[i]
            lists[a].add(b)
            lists[a].add(c)
            lists[b].add(a)
            lists[b].add(c)
            lists[c].add(a)
            lists[c].add(b)
            for facet in [(a,b), (b,c), (a,c)]:
                facet = tuple(sorted(facet)) 
                if facet in boundary_facets:
                    boundary_facets.remove(facet)
                else:
                    boundary_facets.add(facet)
        print("Constructed adjacency lists.")
        print(f"Total boundary facets: {len(boundary_facets)}")
        self._boundary_facets = boundary_facets
        self._adjacency_lists = lists

    @property
    def boundary_facets(self):
        """Get boundary facets
        """

        if self._boundary_facets is None:
            self._init_graph_info()
        return self._boundary_facets

    def _read_nodal_attributes(self, fort13, names=['mannings_n_at_sea_floor']):
        """Read in mannings n and other nodal attributes
        """

        with open(fort13, 'r') as fp:
            fp.readline()
            num_nodes = int(fp.readline())
            num_attrs = int(fp.readline())
            attrs = {}
            for i in range(num_attrs):
                name =  fp.readline().strip()
                unit = fp.readline()
                dim = int(fp.readline())
                if not dim: continue
                default = fp.readline()
                if name not in names: continue
                default = float(default)
                attrs[name] = np.full(num_nodes, default)

            for i in range(num_attrs):
                name = fp.readline().strip()
                num_vals = int(fp.readline())
                if name in names:
                    for j in range(num_vals):
                        node, val = fp.readline().split()
                        attrs[name][int(node)-1] = float(val)
                else:
                    for j in range(num_vals): fp.readline()

            self.nodal_attributes = attrs



    def _read_mesh(self, fort14):
        with open(fort14, 'r') as fp:
            title = fp.readline()
            #NE number of elements, NP number of grid points
            NE,NP=fp.readline().split()
            NE=int(NE)
            NP=int(NP)

            #initiate data structures
            NODENUM=np.zeros(NP, dtype=np.int64)
            LONS=np.zeros(NP)
            LATS=np.zeros(NP)
            DPS=np.zeros(NP)
            ELEMNUM=np.zeros(NE, dtype=np.int64)
            NM = np.zeros((NE,3), dtype=np.int64) #stores connectivity at each element

            #read node information line by line 
            for i in range(NP):
                NODENUM[i], LONS[i], LATS[i], DPS[i] = fp.readline().split()
            #read in connectivity
            for i in range(NE):
                ELEMNUM[i], DUM, NM[i,0],NM[i,1], NM[i,2]=fp.readline().split()

            #(we need to shift nodenum down by 1)
            ELEMNUM=ELEMNUM-1
            NM=NM-1
            NODENUM=NODENUM-1
            self.elemnum = ELEMNUM
            self.nm = NM
            self.nodenum = NODENUM
            self.lons, self.lats, self.depth = LONS, LATS, DPS
            coords = np.column_stack([self.lons, self.lats])
            if not self.cartesian:
                coords = np.deg2rad(coords)    
                if self.projected:
                    # use the equator as a reference point
                    coords[:, 0] *= R*np.cos(np.deg2rad(self.lat0))
                    coords[:, 1] *= R
            self.coords = coords
            # first read open ocean boundary . . .
            NOPE = int(fp.readline().split()[0])
            self.num_open_nodes = NETA = int(fp.readline().split()[0])
            open_node_num = 0
            for i in range(NOPE):
                segment_nodes = int(fp.readline().split()[0])
                segment_node_list = []
                segment_node_index = []
                for j in range(segment_nodes):
                    node = int(fp.readline().split()[0]) - 1
                    segment_node_list.append(node)
                    segment_node_index.append(open_node_num)
                    open_node_num += 1

                self.boundaries["open"].append({
                    "orig_node": segment_node_list,
                    # index into tidal forcing
                    "node_index": segment_node_index,
                    "coords": self.coords[np.array(segment_node_list)].tolist(),
                })
            
            NBOU = int(fp.readline().split()[0])
            NVEL = int(fp.readline().split()[0])
            for i in range(NBOU):
                self._read_boundary_segment(fp)

            # process all levees
            self._finalize_levees()

    def _add_wall_segment(self, segment_node_list):
        """Add a wall boundary segment
        """

        self.boundaries["wall"].append({
                "orig_node": [int(n) for n in segment_node_list],
                "coords": self.coords[np.array(segment_node_list)].tolist(),
            })


    def _read_boundary_segment(self, fp):
        """Read a boundary segment from the open file
        """
        parts = fp.readline().split()
        segment_nodes, boundary_type = int(parts[0]), int(parts[1])
        segment_node_list = [] 
        # another kind of wall boundary
        if boundary_type not in [0, 20, 24, 1, 52, 23]:
            raise RuntimeWarning("Unrecognized boundary type ", boundary_type)

        if boundary_type == 24:
            # levee - if present in mesh treat as wall condition
            levee_front = []
            levee_back = []
            print(f"Parsing levee with {segment_nodes} node pairs.")
            for j in range(segment_nodes):
                parts = fp.readline().split()

                levee_front.append(int(parts[0])-1)
                levee_back.append(int(parts[1])-1)
            # need to figure out how to connect the two states
            head_path = self._connect_nodes(levee_front[0], levee_back[0])
            tail_path = self._connect_nodes(levee_front[-1], levee_back[-1])

            if head_path is not None:
                if tail_path is not None:
                    segment_node_list = levee_front + tail_path + levee_back[::-1] + head_path[::-1]
                    segment_node_list.append(segment_node_list[0])
                else:
                    segment_node_list = levee_front[::-1] + head_path + levee_back
                self._add_levee(segment_node_list)
            elif tail_path is not None:
                self._add_levee(levee_front + tail_path + levee_back[::-1])
            else:
                # two separate wall segments
                self._add_levee(levee_front)
                self._add_levee(levee_back)

        else:
            for j in range(segment_nodes):
                node = int(fp.readline().split()[0]) - 1
                segment_node_list.append(node)
        
            if boundary_type == 1:
                #island boundary, so repeat first node
                segment_node_list.append(segment_node_list[0])
            self._add_wall_segment(segment_node_list)

    def _connect_nodes(self, front, back):
        """Find the path connecting two nodes at the end of a levee
        """

        adj_lists = self.adjacency_lists
        front_neighbors = adj_lists[front]
        back_neighbors = adj_lists[back]
        if back in front_neighbors or front in back_neighbors: return [] # the nodes are connected
        for f in front_neighbors:
            l = adj_lists[f]
            for b in back_neighbors:
                if b in l:
                    print(f"Connected {front} to {back}")
                    return [int(f),int(b)]
        print(f"Could not find path connecting {front} to {back}")
        return None 

    def _add_levee(self, segment):
        """Add a connected portion of a levee
        """

        self._levee_segments.append(segment)

    def _finalize_levees(self):
        """Finalize the levee boundaries
        """

        # Because levees can be conjoined in all sort of horrific ways, we need some fancy
        # post-processing to detect facets with no specified boundary condition
        for segment in self._levee_segments:
            self._add_wall_segment(segment)
        # verify we have no unassigned boundaries
        assigned_boundary_facets = set()
        all_boundary_facets = self.boundary_facets
        for boundary in self.boundaries['open'] + self.boundaries['wall']:
            nodes = boundary['orig_node']
            for i in range(len(nodes)-1):
                facet = tuple(sorted((nodes[i], nodes[i+1])))
                assigned_boundary_facets.add(facet)

        diff = all_boundary_facets - assigned_boundary_facets
        print("Facets with no assigned boundary:", diff)
        print("Assigned facets not actually on boundary:", assigned_boundary_facets-all_boundary_facets)
        print(f"Adding {len(diff)} extra walls.")
        for facet in diff:
            self._add_wall_segment(facet)


    def _read_tides(self, fort15):
        """Read tidal information
        """

        tide_potential = {}
        tide_boundary = {}
        with open(fort15, "r") as fp:
            # Take care of any comments
            lines = [l.split("!")[0] for l in fp.readlines()]
            for i, line in enumerate(lines):
                try:
                    line = line.strip().split()[0]
                except IndexError: continue
                if line in supported_tides:
                    tide = line
                    # check to see what the next line looks like
                    parts = lines[i+1].strip().split()
                    nparts = len(parts)
                    if nparts == 5 and tide not in tide_potential:
                        # read tidal potential information
                        parts = list(map(float, parts))
                        tide_potential[tide] = {
                            "amplitude": parts[0],
                            "frequency": parts[1],
                            "reduction_factor": parts[2],
                            "nodal_factor": parts[3],
                            "equib_arg": parts[4]
                        }
                    elif nparts == 3 and tide not in tide_boundary:
                        # read forcing frequency information
                        parts = list(map(float, parts))
                        tide_boundary[tide] = {
                            "frequency": parts[0],
                            "nodal_factor": parts[1],
                            "equib_arg": parts[2]
                        }
                    elif nparts in [2,4] and tide in tide_boundary:
                        if "amplitude" in tide_boundary[tide]: continue
                        info = {"amplitude": [], "phase": []}
                        for j in range(i+1, i+self.num_open_nodes+1):
                            amp, phase = list(map(float, lines[j].strip().split()))[:2]
                            info["amplitude"].append(amp)
                            info["phase"].append(phase)
                        tide_boundary[tide].update(info)

        self.boundaries["tidal_forcing"] = tide_boundary
        self.boundaries["tidal_potential"] = tide_potential
        
    def _create_mesh(self):
        """Create FEniCSx mesh
        """

        gdim, shape, degree = 2, "triangle", 1
        if use_basix:
            #element = ufl.Mesh(FiniteElement("Lagrange", ufl.Cell(shape), 1, (gdim,), identity_pullback, H1)) 
            #dom = coordinate_element(basix.finite_element.create_element(basix.finite_element.ElementFamily.P, basix.cell.CellType.triangle, 1))
            dom = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        else:
            cell = ufl.Cell(shape, geometric_dimension=gdim)
            dom = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
        
        self.domain = mesh.create_mesh(MPI.COMM_WORLD, self.nm, self.coords, dom)
        V = functionspace(self.domain, ("P", 1))
        dof_coords = self.domain.geometry.x[:, :2]
        inds1 = np.lexsort(dof_coords.T)
        inds2 = np.lexsort(self.coords.T)
        mapping = np.zeros(len(self.coords), dtype=np.int64)
        mapping[inds1] = inds2
        print(np.allclose(dof_coords[:, :2], self.coords[mapping]))
        
        
        self.bathy_func = fem.Function(V)
        self.bathy_func.x.array[:] = self.depth[mapping]
        if 'mannings_n_at_sea_floor' in self.nodal_attributes:
            self.mannings_n_func = fem.Function(V)
            self.mannings_n_func.x.array[:] = self.nodal_attributes['mannings_n_at_sea_floor'][mapping]


    def write_xmdf(self, outfile):
        """Write out mesh as xdmf
        """
        encoding= io.XDMFFile.Encoding.HDF5
        xdmf = io.XDMFFile(self.domain.comm, outfile, "w", encoding=encoding)
        xdmf.write_mesh(self.domain)
        xdmf.write_function(self.bathy_func)
        xdmf.close()

    def write_adios(self, outfile):
        """So Dolfinx can't currently read functions from xdmf files
        
        Conseqeuntly, we need to use an external package, adios4dolfinx, to do it.
        The alternative is to do something like parallel HDF5 . . .
        """

        adios4dolfinx.write_mesh(filename=outfile+"_mesh.bp", mesh=self.domain)
        adios4dolfinx.write_function(filename=outfile+"_depth.bp", u=self.bathy_func)
        if hasattr(self, 'mannings_n_func'):
            adios4dolfinx.write_function(u=self.mannings_n_func, filename=outfile+"_mannings_n.bp")

        with open(outfile+"_boundary.json", "w") as fp:
            json.dump(self.boundaries, fp)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("inputsdir")
    parser.add_argument("outfile")
    parser.add_argument("--cartesian", action="store_true")
    parser.add_argument("--projected", action="store_true")
    args = parser.parse_args()
    mesh = ADCIRCMesh(args.inputsdir, cartesian=args.cartesian, projected=args.projected)
    mesh.write_adios(args.outfile)
