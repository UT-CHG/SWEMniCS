
"""
Custom Newton solver for general nonlinear variational problems.

This was implemented because more control was desired over the Newton iteration than provided by the built-in NonlinearProblem class.
"""

from dolfinx import fem as fe, nls, log,geometry,io,cpp
import dolfinx.fem.petsc as petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix
import numpy.linalg as la
import time
try:
    import cudolfinx as cufem
    from cudolfinx.la import CUDAVector
    have_cuda = True
except ImportError:
    have_cuda = False

def petsc_to_csr(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.size)

class CustomNewtonProblem:

    """An all-in-one class that solves a nonlinear problem. . .
    """
    
    def __init__(self, obj1, solver_parameters={}, cuda=False):
        """initialize the problem
        
        obj1 -- SWEMniCS Solver object
        """
        self.u = obj1.u
        self.F = obj1.F
        self.J = ufl.derivative(self.F, self.u)
        if cuda and not have_cuda:
            raise RuntimeError("Cuda backend not available for Newton Solver without cudolfinx!")

        self.cuda = cuda
        if cuda:
            self.asm = cufem.CUDAAssembler()
            self.residual = cufem.form(self.F)
            self.jacobian = cufem.form(self.J)
        else:
            self.residual = fe.form(self.F)
            self.jacobian = fe.form(self.J)
        self.bcs = obj1.problem.dirichlet_bcs
        self.comm = obj1.problem.mesh.comm

        #relative tolerance for Newton solver
        self.rtol = 1e-5
        #absolute tolerance for Newton solver
        self.atol = 1e-6
        #max iteration number for Newton solver
        self.max_it = 5
        #relaxation parameter for Newton solver
        self.relaxation_parameter = 1.00
        #underlying linear solver
        #default for serial is lu, default for mulitprocessor is gmres
        if self.comm.Get_size() == 1:
            self.ksp_type = "gmres"#preonly
            self.pc_type = "ilu"#lu
        else:
            self.ksp_type = "gmres"
            self.pc_type = "bjacobi"

        for k, v in solver_parameters.items():
            setattr(self, k, v)

        if cuda:
            self.A = self.asm.create_matrix(self.jacobian)
            self.L = self.asm.create_vector(self.residual)
            self.bcs = self.asm.pack_bcs(self.bcs)
        else:
            self.A = petsc.create_matrix(self.jacobian)
            self.L = petsc.create_vector(self.residual)
        self.solver = PETSc.KSP().create(self.comm)

        self.solver.setTolerances(rtol=solver_parameters.get("ksp_rtol",1e-8), atol=solver_parameters.get("ksp_atol", 1e-9), max_it=solver_parameters.get("ksp_max_it", 1000))
        if cuda:
            self.solver.setOperators(self.A.mat)
        else:
            self.solver.setOperators(self.A)
        self.pc = self.solver.getPC()
        # PETSC doesn't support ILU-type pcs on the GPU anymore
        if self.cuda and self.pc_type in ["bjacobi", "ilu"]:
            self.pc_type = "jacobi"
        self.log("pc type", self.pc_type)
        self.pc.setType(self.pc_type)


    def log(self, *msg):
        if self.comm.rank == 0:
            print(*msg)
    
    def solve(self, u):
        """Solve the nonlinear problem at u
        """

        if not hasattr(self, 'dx'):
            self.dx = dx = fe.Function(u._V)
            if self.cuda:
                dx.x.petsc_vec.setType(PETSc.Vec.Type.CUDA)
        else:
            dx = self.dx
        if self.cuda and u.x.petsc_vec.getType() != PETSc.Vec.Type.CUDA:
            u.x.petsc_vec.setType(PETSc.Vec.Type.CUDA)
            self.u_vec = CUDAVector(self.asm._ctx, u.x.petsc_vec)
        if self.cuda:
            # Update all Dirichlet boundary conditions
            self.bcs.update()
        i = 0
        rank = self.comm.rank
        A, L, solver = self.A, self.L, self.solver
        relaxation_parameter = self.relaxation_parameter
        while i < self.max_it:
            # Assemble Jacobian and residual
            self.log("starting newton iteration ", i)
            if self.cuda:
                self.jacobian.to_device()
                self.residual.to_device()
                self.asm.assemble_matrix(self.jacobian, mat=A)
                A.assemble()
                self.asm.assemble_vector(self.residual, L)
                rhs = L.vector
                rhs.scale(-1)
                self.asm.apply_lifting(L, [self.jacobian], [self.bcs], x0=[self.u_vec])
                self.asm.set_bc(L, self.bcs, u.function_space, x0=self.u_vec, scale=1.0)
            else:
                with L.localForm() as loc_L:
                    loc_L.set(0)
                A.zeroEntries()
                petsc.assemble_matrix(A, self.jacobian, bcs=self.bcs)
                A.assemble()
                petsc.assemble_vector(L, self.residual)
                L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                L.scale(-1)
                # Compute b - J(u_D-u_(i-1))
                petsc.apply_lifting(L, [self.jacobian], [self.bcs], x0=[u.x.petsc_vec], alpha=1)
                # Set dx|_bc = u_{i-1}-u_D
                petsc.set_bc(L, self.bcs, u.x.petsc_vec, 1.0)
                L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
                rhs = L
            self.log("Residual norm", rhs.norm(0))

            # Solve linear problem
            solver.solve(rhs, dx.x.petsc_vec)
            
            dx.x.scatter_forward()
            self.log(f"linear solver convergence {solver.getConvergedReason()}" +
                    f", iterations {solver.getIterationNumber()}, resid norm {solver.getResidualNorm()}")
            if solver.getConvergedReason() == -9:
                raise RuntimeError("Linear Solver failed due to nans or infs!!!!")
            # Update u_{i+1} = u_i + delta x_i
            u.x.petsc_vec.axpy(relaxation_parameter, dx.x.petsc_vec)
            if self.cuda:
                # Ensure host-side values of u match device-side values
                nlocal = len(u.x.petsc_vec.array)
                # don't copy ghosts as we can't explicitly update them
                u.x.array[:nlocal] = u.x.petsc_vec.array

            u.x.scatter_forward() 
            i += 1
            
            dx_norm = dx.x.petsc_vec.norm(0)
            if i == 1:
                self.dx_0_norm = dx_norm
                self.log('dx_0 norm,',self.dx_0_norm)
            
            if self.dx_0_norm > 1e-8:
                correction_norm = dx_norm/self.dx_0_norm
            else:
                correction_norm = dx_norm

            self.log(f"Netwon Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < self.atol:
                break
            if hasattr(self, 'reduction_it'):
                if i  and i %self.reduction_it == 0:
                    self.log("Still haven't converged. Reducing relax param")
                    relaxation_parameter /= 2

