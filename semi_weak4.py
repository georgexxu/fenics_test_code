from fenics import *
import time
import numpy as np
import math
from scipy import stats
#import matplotlib.pyplot as plt

def computeArea(eta): # robust in non-parallel mode
    '''
    robust in non-parallel mode
    time consuming, when interpolate on 120*120*120, takes 93s, yet solving only takes 20s
    the most time consuming step is interpolate
    '''
    #eta = interpolate(eta,V)
    #eta = project(eta,V)
    values = eta.vector().get_local()
    values = np.where(values > - DOLFIN_EPS,1,values)
    values = np.where(values != 1, 0, values)
    Area = 256.0 * 256.0 * np.sum(values)/float(len(values))
    return Area

# with open('semi_amg_2D_CPU.txt', 'w') as fCPU:
#     fCPU.write("This file is used to record the CPU time\n")
# with open('semi_cg_2D_slope_time_1.txt', 'w') as f_slope:
#     f_slope.write("This file is used to record the slope and relative error\n")
# exact_slope = -2*math.pi

rank = MPI.rank(MPI.comm_world)
for grid_point in [20000]:
# for grid_point in [2048]:
    for dt in [1]:
    # for dt in [1.0]:
        
        mesh = RectangleMesh(Point(-128,-128),Point(128,128),grid_point,grid_point)

        T = 0.1
        print_ferquence = int(50/dt)

        P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
        V1 = FunctionSpace(mesh,P1)

        eta = TrialFunction(V1)
        v = TestFunction(V1)
        eta_n = Function(V1)

        #initial condition
        eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
        #eta_0 = project(eta_0,V1)
        eta_n.assign(eta_0)
        #eta.assign(eta_0)

        #weak form
        #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx

        a = (eta)/dt*v*dx + dot(grad(eta),grad(v))*dx 

        L = (eta_n)/dt*v*dx + (- eta_n * ( eta_n-1 ) * ( eta_n+1 ) * v * dx)

        p = dot(grad(eta),grad(v))*dx 

        eta = Function(V1)


        t = 0 
        counter = 0

        Rs = [] 
        ts = []


        a1 = time.time()

        parameters["linear_algebra_backend"] = "PETSc"
        
        krylov_method="cg"
        precond="hypre_amg"
        #precond="none"    
        solver = KrylovSolver(krylov_method, precond)
        #solver = KrylovSolver(krylov_method)

        solver.parameters["relative_tolerance"] = 1.0e-10
        solver.parameters["absolute_tolerance"] = 1.0e-10
        solver.parameters["monitor_convergence"] = True
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters['nonzero_initial_guess'] = True

        A, b = assemble_system(a, L)


        start_time = time.time()
        while t < T: 
            t += dt
            print("===========================> current time: ",t)

            #solver.set_operator(A)  
            b = assemble(L)
            w_time = time.time()
            cpu_time = time.process_time()
            solver.solve(A, eta.vector(), b)
            if rank == 0: 
                print("Wall time is: ",time.time()-w_time)
                print("Process time is: ",time.process_time()-cpu_time)
            eta_n.assign(eta)
