"""Handling of boundary conditions and facet tags.
"""

from dolfinx import fem as fe
import dolfinx.mesh as mesh
from ufl import (div, as_tensor, as_vector, inner, dx,ds,Measure)
import numpy as np

class BoundaryCondition:
    """A class describing a boundary condition.
    """

    def __init__(self, type, marker, forcing_func=None, V=None, bound_func=None, facet_tag=None):
        self._type = type
        self._func = forcing_func

        if type == "Open":
            if bound_func is not None:
                #should work for CG and DG
                dofs = fe.locate_dofs_geometrical((V, V.collapse()[0]), bound_func)[0]
            elif facet_tag is not None:
                facets = facet_tag.find(marker)
                fdim = 1 #hardcoded for 2d
                #only works for CG
                dofs = fe.locate_dofs_topological((V, V.collapse()[0]), fdim, facets)[0]
            self._bc = fe.dirichletbc(forcing_func, dofs)
            self._dofs = dofs
            self._marker = marker

        elif type == "Wall" or type == "OF" or type =="Flux":
                #Since Wall condition is enforced weakly, only the marker contains info
                self._bc = []
                self._marker = marker
                self._dofs = np.array([])
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

    @property
    def dofs(self):
        return self._dofs

    @property
    def marker(self):
        return self._marker

    @property
    def func(self):
        return self._func

def MarkBoundary(domain,boundaries):

    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1

    #need to restrict to boundary facets, works for now byt not sure if works in general
    #WARNING: Not sure if this works for DG yet, but at least for tidal case it does
    boundary_facets=mesh.locate_entities_boundary(domain,fdim,lambda x:np.full(x.shape[1],True,dtype=bool))

    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facets = np.intersect1d(facets,boundary_facets)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    return facet_markers,facet_tag
