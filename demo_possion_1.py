from dolfin import *
import time

rank = MPI.rank(MPI.comm_world)

# Create mesh and define function space
# mesh = UnitSquareMesh(5000,5000)
nx = 5000
#mesh = RectangleMesh(Point(0, 0), Point(100, 100), nx, nx)
mesh = UnitSquareMesh(nx,nx)
V = FunctionSpace(mesh, "Lagrange", 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1 - DOLFIN_EPS
# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds


parameters["linear_algebra_backend"] = "PETSc"
krylov_method="cg"
precond="hypre_amg"   
solver = KrylovSolver(krylov_method, precond)
solver.parameters["relative_tolerance"] = 1.0e-10
solver.parameters["absolute_tolerance"] = 1.0e-10
solver.parameters["monitor_convergence"] = True
solver.parameters["maximum_iterations"] = 1000

A, b = assemble_system(a, L,bc)
u = Function(V)

cpu_time = time.process_time()
solver.solve(A, u.vector(), b)
if rank == 0: 

	print("Process time is: ",time.process_time()-cpu_time)

