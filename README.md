# SWEMnics

This package allows for solving the shallow water equations with FEniCSx (0.9.0) with a wide variety of numerical methods.

> [!NOTE]  
> Now works with fenicsx version 0.9.0

# Description

The primary components of the package are a set of solvers, and a problem set. Example usage for both can be found in the examples folder. Both problems and solvers can be extended to support customized applications and/or numerical methods. Details on numerics can also be found in the [SWEMniCS paper](https://www.nature.com/articles/s44304-024-00036-5) 

# Installation

First, clone or download this git repository, then use either

**Conda**

The recommended way is to first set up a Python environment with fenicsx via conda/mamba:

```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista scipy matplotlib h5py adios4dolfinx
```

**Docker**

Alternatively, run a DOLFINx docker container, for instance:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m ghcr.io/fenics/dolfinx/dolfinx:stable
```

Then after using either conda or docker to set up your environment run

```bash
python3 -m pip install --no-build-isolation -e .
```

# Verification of installation

To run a tidal flow example in the examples folder:

```bash
python examples/tidal.py
```
