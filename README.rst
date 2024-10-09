========
SWEMnics
========


This package allows for solving the shallow water equations with FEniCSx (0.8.0) with a wide variety of numerical methods.


Description
===========

The primary components of the package are a set of solvers, and a problem set. Example usage for both can be found in the examples folder. Both problems and solvers can be extended to support customized applications and/or numerical methods.


Installation
============

The recommended way is to first set up a Python environment with fenicsx via conda:\
conda create -n fenicsx-env\
conda activate fenicsx-env\
conda install -c conda-forge fenics-dolfinx mpich pyvista\

There are a couple of extra dependencies within the SWEMniCS repo:
conda install conda-forge::adios2\
pip install matplotlib\
pip install scipy\
pip install h5py\
python -m pip install adios4dolfinx[test]\

To install SWEMniCS locally, navigate to the main folder of the cloned repo and simply:\
pip install ../SWEMniCS\

To run a tidal flow example in the examples folder:\
python examples/tidal.py
