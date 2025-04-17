
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

def petsc_to_csr(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.size)

class CustomNewtonProblem:

    """An all-in-one class that solves a nonlinear problem. . .
    """
    
    def __init__(self, obj1,solver_parameters={}):
        """initialize the problem
        
        F -- Ufl form
        """
        self.u = obj1.u
        self.F = obj1.F
        self.residual = fe.form(self.F)

        self.J = ufl.derivative(self.F, self.u)
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
            print("serial run")
            self.ksp_type = "gmres"#preonly
            self.pc_type = "ilu"#lu
        else:
            self.ksp_type = "gmres"
            self.pc_type = "bjacobi"

        for k, v in solver_parameters.items():
            setattr(self, k, v)

        self.A = petsc.create_matrix(self.jacobian)
        self.L = petsc.create_vector(self.residual)
        self.solver = PETSc.KSP().create(self.comm)

        self.solver.setTolerances(rtol=solver_parameters.get("ksp_rtol",1e-8), atol=solver_parameters.get("ksp_atol", 1e-9), max_it=solver_parameters.get("ksp_max_it", 1000))
        self.solver.setOperators(self.A)
        self.solver.setErrorIfNotConverged(solver_parameters.get("ksp_ErrorIfNotConverged",True))
        
        if self.pc_type == 'element_block':
            self.pc = ElementBlockPreconditioner(self.A, obj1.problem.mesh)
        else:
            self.pc = self.solver.getPC()
            self.pc.setType(self.pc_type)

        #for tangent linear model work
        self.make_tangent = obj1.make_tangent
        if (self.make_tangent):
            self.F_no_dt = obj1.F_no_dt
            #self.tangent_form = fe.form(self.F_no_dt)
            self.tangent_J = ufl.derivative(self.F_no_dt, self.u)
            self.tangent_jacobian = fe.form(self.tangent_J)
            self.A_tangent = petsc.create_matrix(self.tangent_jacobian)


    def log(self, *msg):
        if self.comm.rank == 0:
            print(*msg)
    
    def solve(self, u, max_it=5):
        """Solve the nonlinear problem at u
        """

        dx = fe.Function(u._V)
        i = 0
        rank = self.comm.rank
        A, L, solver = self.A, self.L, self.solver
        relaxation_parameter = self.relaxation_parameter
        while i < self.max_it:
            # Assemble Jacobian and residual
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            petsc.assemble_matrix(A, self.jacobian, bcs=self.bcs)
            A.assemble()
            petsc.assemble_vector(L, self.residual)
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            L.scale(-1)
            # Compute b - J(u_D-u_(i-1))
            #080
            #petsc.apply_lifting(L, [self.jacobian], [self.bcs], x0=[u.vector], scale=1)
            #090
            petsc.apply_lifting(L, [self.jacobian], [self.bcs], x0=[u.x.petsc_vec], alpha=1)
            # Set dx|_bc = u_{i-1}-u_D
            #080
            #petsc.set_bc(L, self.bcs, u.vector, 1.0)
            #090
            petsc.set_bc(L, self.bcs, u.x.petsc_vec, 1.0)
            L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            self.log("Residual norm", L.norm(0))
            # Solve linear problem
            if self.pc_type == 'element_block':
                #print("A cond num", la.cond(petsc_to_csr(A).todense()))
                new_A, new_rhs = self.pc.precondition(L)
                #print("new_A cond num", la.cond(petsc_to_csr(new_A).todense()))
                #solver.reset()
                #print(solver.getPC().getType())
                #raise ValueError()
                solver = PETSc.KSP().create(self.comm)
                solver.setType("gmres")
                solver.setTolerances(rtol=1e-8, atol=1e-9)
                solver.getPC().setType("mat")
                solver.setOperators(A, self.pc.mat)
                #start = time.time()
                #080
                #solver.solve(L, dx.vector)
                #090
                solver.solve(L, dx.x.petsc_vec)
                #print("solved in ", time.time()-start)
            else:
                #start = time.time()
                #print("pc type", solver.getPC().getType())
                #080
                #solver.solve(L, dx.vector)
                #090
                solver.solve(L, dx.x.petsc_vec)
                #print("solved in ", time.time()-start)
            
            dx.x.scatter_forward()
            self.log(f"linear solver convergence {solver.getConvergedReason()}" +
                    f", iterations {solver.getIterationNumber()}, resid norm {solver.getResidualNorm()}")
            if solver.getConvergedReason() == -9:
                raise RuntimeError("Linear Solver failed due to nans or infs!!!!")
            # Update u_{i+1} = u_i + delta x_i
            #not working in parallel?
            u.x.array[:] += relaxation_parameter*dx.x.array[:]
            
            i += 1
            
            if i == 1:
                #080
                #self.dx_0_norm = dx.vector.norm(0)
                #090
                self.dx_0_norm = dx.x.petsc_vec.norm(0)
                self.log('dx_0 norm,',self.dx_0_norm)


            #this is relative but breaks in parallel?
            #print('dx before', dx.vector.getArray())
            if self.dx_0_norm > 1e-8:
                dx.x.array[:] = np.array(dx.x.array[:]/self.dx_0_norm)
            #why wont this update unless I call it??
            #dx.vector.update()
            #080
            #dx.vector.assemble()
            #090
            dx.x.petsc_vec.assemble()
            #print('dx after', dx.vector.getArray())
            
            # Compute norm of update
            #080
            #correction_norm = dx.vector.norm(0)
            #090
            correction_norm = dx.x.petsc_vec.norm(0)

            self.log(f"Netwon Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < self.atol:
                break
            if hasattr(self, 'reduction_it'):
                if i  and i %self.reduction_it == 0:
                    self.log("Still haven't converged. Reducing relax param")
                    relaxation_parameter /= 2

            #print(A.getValuesCSR())
        #print(L.getArray())
        #print(L.getArray().size)
        #print(u.x.array[:])
    def form_tangent_mat(self):
        self.A_tangent.zeroEntries()
        petsc.assemble_matrix(self.A_tangent, self.tangent_jacobian, bcs=self.bcs)
        self.A_tangent.assemble()
        return self.A_tangent


class ElementBlockPreconditioner:

    def __init__(self, A, mesh):
        """Initialize the preconditioner from the 
        """

        dim = mesh.topology.dim
        num_cells = mesh.topology.index_map(dim).size_local
        print("local num cells", num_cells)
        block_size = A.size[0] // num_cells
        print("block size", block_size)
        self.block_size = block_size
        self.A = A
        mat = PETSc.Mat()
        mat.createAIJ((A.size[0], A.size[0]), nnz=np.full(A.size[0], block_size, dtype=np.int32), comm=mesh.comm)
        mat.setUp()
        mat.setBlockSize(block_size)
        self.mat = mat

    def precondition(self, rhs):
        """Apply the block preconditioner to A and the right hand side

        returns P^-1 * A, P^-1 * rhs
        """

        old_block_size = self.A.getBlockSize()
        self.A.setBlockSize(self.block_size)
        inv = self.A.invertBlockDiagonal()
        self.A.setBlockSize(old_block_size)
        start_ind, stop_ind = self.mat.owner_range
        block_inds = np.arange(start_ind//self.block_size, stop_ind//self.block_size+1)
        block_inds = block_inds.astype(np.int32)
        self.mat.setValuesBlockedCSR(block_inds, block_inds[:-1], inv)
        #for i in range(len(inv)):
        #    mat.setValuesBlocked(block_inds[i:i+1], block_inds[i:i+1], inv[i])
        self.mat.assemble()
        new_rhs = self.mat.createVecRight()
        self.mat.mult(rhs, new_rhs)
        return self.mat.matMult(self.A), new_rhs


class NewtonSolver:

    def __init__(self, obj1,
        #u_init = lambda x: np.ones(x.shape),
        solver_parameters={}
        ):

        """Solve the equation and save the result in u_sol
        """

        prob = petsc.NonlinearProblem(obj1.F, obj1.u, bcs=obj1.problem.get_bcs())

        # the problem appears to be that the residual is humongous. . .
        res = fe.form(obj1.F)
        test_res = petsc.create_vector(res)
        petsc.assemble_vector(test_res, res)
        #print(test_res.getArray())

        self.solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, prob)
        for k, v in solver_parameters.items():
            setattr(self.solver, k, v)
        self.solver.report=True
        self.solver.convergence_criterion = "incremental"
        self.solver.error_on_nonconvergence = False
        log.set_log_level(log.LogLevel.INFO)
        ksp = self.solver.krylov_solver
        

        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        viewer = PETSc.Viewer().createASCII("default_output.txt")
        ksp.view(viewer)

        #ksp.setType("preonly")
        #pc = ksp.getPC()
        #pc.setType("lu")
        #pc.setFactorSolverType("mumps")
        #pc.setFactorSetUpSolverType()

        opts[f"{option_prefix}ksp_type"] = "preonly"
        #opts[f"{option_prefix}pc_type"] = "bjacobi"
        #opts[f"{option_prefix}pc_factor_mat_solver_type"] = "ilu"
        
        #option_prefix = ksp.getOptionsPrefix()
        #opts[f"{option_prefix}ksp_type"] = "gmres"#"preonly"#"gmres"#"richardson"#"preonly"#"cg"
        opts[f"{option_prefix}pc_type"] = "lu"#"bjacobi"#"none"#"lu"#"gamg"
        
        opts[f"{option_prefix}pc_factor_solver_type"] = "mumps"
        #pc = ksp.getPC()
        #pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # For pressure nullspace
        #pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # For pressure nullspace

        ksp.setFromOptions()



        viewer = PETSc.Viewer().createASCII("linear_output.txt")
        ksp.view(viewer)
        
        solver_output = open("linear_output.txt", "r")
        for line in solver_output.readlines():
            print(line)
        #print(self.u.vector.getArray())
    def solve(self,u):
        #print('before Newton', u.x.array[:])
        r = self.solver.solve(u)
        #print(self.solver.A.getValuesCSR())
        #print(self.solver.b.getArray())
        #print(u.x.array[:])
