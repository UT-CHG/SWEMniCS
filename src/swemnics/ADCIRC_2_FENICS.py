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

        if os.path.exists(inputsdir+"/fort.13"):
            self._read_nodal_attributes(inputsdir+"/fort.13")
        else:
            self.nodal_attributes = {}

        self._read_mesh(inputsdir+"/fort.14")
        self._read_tides(inputsdir+"/fort.15")
        self._create_mesh()

    def _read_nodal_attributes(self, fort13):
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
                fp.readline()
                default = float(fp.readline())
                attrs[name] = np.full(num_nodes, default)

            for i in range(num_attrs):
                name = fp.readline().strip()
                num_vals = int(fp.readline())
                for j in range(num_vals):
                    node, val = fp.readline().split()
                    attrs[name][int(node)-1] = float(val)

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
            ELEMNUM=np.zeros(NE)
            NM = np.zeros((NE,3)) #stores connectivity at each element

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
            # we can't directly store boundary information
            # However, we can store it indirectly with a functino
            boundary_info = np.zeros(NP)
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
                    # use -1 for open ocean
                    boundary_info[node] = -1
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
                parts = fp.readline().split()
                segment_nodes, boundary_type = int(parts[0]), int(parts[1])
                segment_node_list = []
                # another kind of wall boundary
                if boundary_type == 20:
                    boundary_type = 0

                for j in range(segment_nodes):
                    node = int(fp.readline().split()[0]) - 1
                    # use -1 for open ocean
                    boundary_info[node] = boundary_type+1
                    segment_node_list.append(node)
                
                if boundary_type == 0: pass
                elif boundary_type == 1:
                    #island boundary, so repeat first node
                    segment_node_list.append(segment_node_list[0])
                else:
                    raise RuntimeWarning("Unrecognized boundary type ", boundary_type)

                # TODO handle other kinds of BCs . . .
                self.boundaries["wall"].append({
                        "orig_node": segment_node_list,
                        "coords": self.coords[np.array(segment_node_list)].tolist(),
                    })
            
        self.boundary_info = boundary_info
        
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
                    if nparts == 5 and tide not in tidal_potential:
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
                    elif nparts == 2 and tide in tide_boundary:
                        if "amplitude" in tide_boundary[tide]: continue
                        info = {"amplitude": [], "phase": []}
                        for j in range(i+1, i+self.num_open_nodes+1):
                            amp, phase = list(map(float, lines[j].strip().split()))
                            info["amplitude"].append(amp)
                            info["phase"].append(phase)
                        tide_boundary[tide].update(info)

        self.boundaries["tidal_forcing"] = tide_boundary
        self.boundaries["tidal_potential"] = tide_potential
        
    def _create_mesh(self):
        """Create FEniCSx mesh
        """

        gdim, shape, degree = 2, "triangle", 1
        cell = ufl.Cell(shape, geometric_dimension=gdim)
        element = ufl.VectorElement("Lagrange", cell, degree)
        
        self.domain = mesh.create_mesh(MPI.COMM_WORLD, self.nm, self.coords, ufl.Mesh(element))
        V = fem.FunctionSpace(self.domain, ("P", 1))
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

        adios4dolfinx.write_mesh(self.domain, outfile+"_mesh.bp", "BP4")
        adios4dolfinx.write_function(self.bathy_func, outfile+"_depth.bp", "BP4")
        if hasattr(self, 'mannings_n_func'):
            adios4dolfinx.write_function(self.mannings_n_func, outfile+"_mannings_n.bp", "BP4")

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
