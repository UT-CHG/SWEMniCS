import h5py
from dolfinx import fem as fe
from swemnics.constants import R
import numpy as np

"""
Classes for describing meteorological forcing.
@author Benjamin Pachev <benjaminpachev@gmail.com>
@author Mark Loveland <markloveland@utexas.edu>
Computation Hydraulics Group
The Oden Intitute for Computational Engineering and Sciences
The University of Texas at Austin
"""


class GriddedForcing:
    """Supports meteorological forcing on a regular lat/lon grid
    """

    def __init__(self, filename, lat0=35):
        """Initialize the forcing
        """
        
        with h5py.File(filename, "r") as ds:
            self.lat = ds["latitude"][:]
            self.lon = ds["longitude"][:]
            self._windx = ds["windx"][:]
            self._windy = ds["windy"][:]
            self._pressure = ds["pressure"][:]
            self.time = ds["time"][:]

        # check dimensions
        nt, nlat, nlon = len(self.time), len(self.lat), len(self.lon)
        shape = (nt, nlat, nlon)
        print(self._windx.shape, self._windy.shape, self._pressure.shape)
        assert self._windx.shape == shape
        assert self._windy.shape == shape
        assert self._pressure.shape == shape
        self.lat0 = lat0

    def set_V(self, V):
        """Set the FunctionSpace to be used
        """

        self._V = V
        self.windx = fe.Function(V)
        self.windy = fe.Function(V)
        self.pressure = fe.Function(V)
        self.coords = V.tabulate_dof_coordinates()[:, :2]
        # reverse projection. . .
        self.coords = np.rad2deg(self.coords/np.array([[R * np.cos(np.deg2rad(self.lat0)), R]]))
        # determine insertion indices
        lon_inds = np.searchsorted(self.lon, self.coords[:, 0])
        lat_inds = np.searchsorted(self.lat, self.coords[:, 1])
        lower_lon_inds = lon_inds-1
        lower_lat_inds = lat_inds-1
        # ensure all indices are within bounds
        lon_inds = np.clip(lon_inds, 0, len(self.lon)-1)
        lower_lon_inds = np.clip(lower_lon_inds, 0, len(self.lon)-1)
        lat_inds = np.clip(lat_inds, 0, len(self.lat)-1)
        lower_lat_inds = np.clip(lower_lat_inds, 0, len(self.lat)-1)

        # assume regular spacing, for now
        lon_cellsize = self.lon[1]-self.lon[0]
        lat_cellsize = self.lat[1]-self.lat[0]
        # compute interpolation weights
        self.lon_pos = (self.coords[:, 0] - self.lon[lower_lon_inds]) / lon_cellsize
        self.lat_pos = (self.coords[:, 1] - self.lat[lower_lat_inds]) / lat_cellsize
        self.lon_inds, self.lower_lon_inds = lon_inds, lower_lon_inds
        self.lat_inds, self.lower_lat_inds = lat_inds, lower_lat_inds

    
    def evaluate(self, t):
        """Evaluate wind and pressure at a specified time t
        """

        t_ind = np.searchsorted(self.time, t)
        time_resolution = self.time[1] - self.time[0]
        last_ind = t_ind - 1
        if last_ind < 0: last_ind = 0
        if t_ind >= len(self.time): t_ind = len(self.time) - 1
        t_pos = (t - self.time[last_ind]) / (time_resolution)

        windx = t_pos * self._windx[last_ind] + (1 - t_pos) * self._windx[t_ind]
        windy = t_pos * self._windy[last_ind] + (1 - t_pos) * self._windy[t_ind]
        pressure = t_pos * self._pressure[last_ind] + (1 - t_pos) * self._pressure[t_ind]
        
        self.windx.x.array[:] = (
            self.lon_pos * self.lat_pos * windx[self.lower_lat_inds, self.lower_lon_inds] +
            self.lon_pos * (1 - self.lat_pos) * windx[self.lat_inds, self.lower_lon_inds] +
            (1 - self.lon_pos) * self.lat_pos * windx[self.lower_lat_inds, self.lon_inds] +
            (1 - self.lon_pos) * (1 - self.lat_pos) * windx[self.lat_inds, self.lon_inds]
        )

        self.windy.x.array[:] = (
            self.lon_pos * self.lat_pos * windy[self.lower_lat_inds, self.lower_lon_inds] +
            self.lon_pos * (1 - self.lat_pos) * windy[self.lat_inds, self.lower_lon_inds] +
            (1 - self.lon_pos) * self.lat_pos * windy[self.lower_lat_inds, self.lon_inds] +
            (1 - self.lon_pos) * (1 - self.lat_pos) * windy[self.lat_inds, self.lon_inds]
        )

        # convert from mBar to pascals
        self.pressure.x.array[:] = 100 * (
            self.lon_pos * self.lat_pos * pressure[self.lower_lat_inds, self.lower_lon_inds] +
            self.lon_pos * (1 - self.lat_pos) * pressure[self.lat_inds, self.lower_lon_inds] +
            (1 - self.lon_pos) * self.lat_pos * pressure[self.lower_lat_inds, self.lon_inds] +
            (1 - self.lon_pos) * (1 - self.lat_pos) * pressure[self.lat_inds, self.lon_inds]
        )

        
