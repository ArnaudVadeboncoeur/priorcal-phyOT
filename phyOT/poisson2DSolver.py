import os

import jax
import jax.numpy as jnp
import jaxopt
import sys
from functools import partial
from jaxinterp2d import interp2d
import time
from phyOT.utils import *

def shift(u, ipm=None, jpm=None):
    if ipm == 'p': u = jnp.pad(u[1:, :], [[0,1], [0,0]])
    elif ipm =='m': u = jnp.pad(u[:-1, :], [[1,0],[0,0]])
    
    if jpm == 'p': u = jnp.pad(u[:, 1:], [[0,0], [0,1]])
    elif jpm == 'm': u = jnp.pad(u[:, :-1], [[0,0], [1,0]])

    return u

class Poisson2DSolver():
    '''
        class with methods for solving 2D Poisson from \nabla \cdpt ( a_z(u, x, y) * \nabla u) - f = 0
        using a Finite Difference implementation of the weak for
        equiv. to Finite Elements with linear basis functions.

    '''
    def __init__(self, nx, verbose=True):

        self.nx = nx
        self.grid = self.make_grid(nx, nx) #[X.shape(nx, ny), Y.shape(nx, ny)] <==> [nx, ny, 2]
        self.dx = self.grid[0, 0,1] - self.grid[0,0,0]
        print(f'dx = {self.dx:.5f}')
        print(1./(self.nx - 1))
        self.dx = 1./(self.nx - 1)
        self.u_init = self.grid[0]*0.
        self.sln_shape = self.grid.shape
        self.solverChoice = 'root' #! 'lm' || 'root'
        self.residuals_info = False
        self.device = 'cuda'
        self.verbose = verbose
        self.residual_evaluator = self.residual_WeakForm_Hat_shift_indexing

    def make_grid(self, nx, ny):
        x = jnp.linspace(0., 1., nx)
        y = jnp.linspace(0., 1., ny)
        XY = jnp.stack(jnp.meshgrid(x, y, indexing='xy'))
        XY = XY.at[1,:,:].set( jnp.flip(XY[1,...], axis=0))
        return XY
    
    def interp2d_wrapped(self, u, x):
        ''' 
        u as [nx, nx] grid
        x as [n, 2]   array
        '''
        return interp2d(x[:,0], x[:,1], self.grid[0][0,:], jnp.flip(self.grid[1][:,0]), u)
        
    def boundary_func(self, u, grid):
        return u * jnp.sin(jnp.pi * grid[0]) * jnp.sin(jnp.pi*grid[1])
        
    def residual(self, u, z, f, with_bc = False, rhs_lhs = None):

        if rhs_lhs is not None:
            u = u.reshape(self.nx-2, self.nx-2)
            u = jnp.pad(u, [[1,1], [1,1]])
            f = f.reshape(self.nx-2, self.nx-2)
            f = jnp.pad(f, [[1,1], [1,1]])

        res_square = self.residual_evaluator(u.reshape(self.nx, self.nx), z.reshape(self.nx, self.nx), 
                                             f.reshape(self.nx, self.nx), with_bc = with_bc, rhs_lhs=rhs_lhs)
        return res_square.reshape(-1,)

    def residual_WeakForm_Hat_shift_indexing(self, u, z, f, with_bc = False, rhs_lhs = None):

        # ij   = [1:-1, 1:-1]
        # ipj  = [2:, 1:-1]
        # imj  = [:-2, 1:-1]
        # ijm  = [1:-1, :-2]
        # ijp  = [1:-1, 2:]
        # impm = [:-2, :-2]
        # ipjp = [2:, 2:]

        #Dirichlet BC => ignore boundary test nodes as set to zero
        r = jnp.zeros(u[1:-1, 1:-1].shape)
        # r = - r_x1 - r_x2 + rf

        r_x1 = jnp.zeros(u[1:-1, 1:-1].shape)

        r_x1 = r_x1.at[:, :].add( (z[:-2, :-2] + z[1:-1, :-2]) * ( u[1:-1, 1:-1] - u[1:-1, :-2] ) ) 
        r_x1 = r_x1.at[:, :].add(-(z[1:-1, 1:-1] + z[:-2, 1:-1]) * ( u[1:-1, 2:] - u[1:-1, 1:-1] ) )  

        r -= 0.5 * r_x1

        r_x2 = jnp.zeros(u[1:-1, 1:-1].shape)
        r_x2 = r_x2.at[:, :].add(+(z[1:-1, :-2] + z[1:-1, 1:-1]) * ( u[1:-1, 1:-1] - u[2:, 1:-1] ) )
        r_x2 = r_x2.at[:, :].add(-(z[:-2, 1:-1] + z[:-2, :-2]) * ( u[:-2, 1:-1] - u[1:-1, 1:-1] ) )

        r -= 0.5 * r_x2

        rf = jnp.zeros(u[1:-1, 1:-1].shape)
        rf = rf.at[:, :].add(self.dx**2. / 9. * ( 3 * f[1:-1, 1:-1] + f[:-2, :-2] + f[1:-1, :-2] +
                                  f[2:, 1:-1] + f[2:, 2:] + f[1:-1, 2:] + f[:-2, 1:-1]))

        r += rf

        if rhs_lhs is not None:
            if rhs_lhs == 'lhs':
                return (-0.5 * r_x1 - 0.5 * r_x2).reshape(-1,)
            elif rhs_lhs == 'rhs':
                return rf.reshape(-1,)

        if with_bc ==  False:
            return r
        
        else:
            # boundary residual
            rb_1   = (0. - u[ :1, 1:-1]).reshape(1,-1)
            rb_2   = (0. - u[-1:, 1:-1]).reshape(1,-1)
            rb_3   = (0. - u[:, :1] ).reshape(-1,1)
            rb_4   = (0. - u[:, -1:]).reshape(-1,1)

            r = jnp.concatenate((rb_1, r, rb_2), axis=0)
            r = jnp.concatenate((rb_3, r, rb_4), axis=1)

            return r
        

    def solve(self, u_init, z, f):

        with jax.default_device(jax.devices(self.device)[0]):

            residual_func = lambda u, z, f: self.residual(u, z, f, with_bc=True)

            print('SOLVER CHOICE: ', self.solverChoice)

            if self.solverChoice == 'root':

                opt = jaxopt.ScipyRootFinding(
                        optimality_fun = residual_func,
                        method='hybr',
                        tol = 1e-5,
                        jit=True, implicit_diff_solve=jaxopt.linear_solve.solve_cg, 
                        has_aux=False,
                        options={'disp':True, 'xtol':1e-5, 'maxfev':200}, use_jacrev=False)
                               
            elif self.solverChoice == 'lm':
                opt = jaxopt.LevenbergMarquardt(residual_func, maxiter=100, damping_parameter=1e-06,
                        stop_criterion='madsen-nielsen',
                        # stop_criterion='grad-l2-norm',
                        tol=0.0001, xtol=0.0001, gtol=0.0001, 
                        geodesic=True, 
                        verbose=self.verbose, jac_fun=None,
                        solver = jaxopt.linear_solve.solve_cg,
                        materialize_jac=False,
                        implicit_diff=True, 
                        implicit_diff_solve=jaxopt.linear_solve.solve_cg,
                        has_aux=False, jit='auto', unroll='auto')
            
            if self.solverChoice == 'root' or self.solverChoice == 'lm':
                opt_sln, state = opt.run(u_init, z, f)
                # print('L2 opt error', opt.l2_optimality_error(opt_sln, self.grid, z))

            elif self.solverChoice == 'cg' or self.solverChoice == 'gmres' or self.solverChoice == 'bicgstab':
                if self.solverChoice == 'cg': linearsolver = jaxopt.linear_solve.solve_cg
                elif self.solverChoice == 'gmres': linearsolver = jaxopt.linear_solve.solve_gmres
                elif self.solverChoice == 'bicgstab': linearsolver = jaxopt.linear_solve.solve_bicgstab
                print(f'USING Linear Solver: {self.solverChoice}')
                start_time = time.time()
                rhs = self.residual(jnp.zeros((self.nx-2, self.nx-2)).reshape(-1,), z, f.reshape(self.nx, self.nx)[1:-1, 1:-1].reshape(-1,), with_bc=False, rhs_lhs='rhs')
                print('rhs.shape', rhs.shape)
                res_bind = lambda u : self.residual(u, z.reshape(-1,), f.reshape(self.nx, self.nx)[1:-1, 1:-1].reshape(-1,), with_bc=False, rhs_lhs='lhs')
                res_jit = jax.jit(res_bind)
                opt_sln = linearsolver(res_jit, -rhs, tol=1e-5)
                opt_sln = jnp.pad(opt_sln.reshape(self.nx-2, self.nx-2), [[1,1], [1,1]])
                print('time tken sln:', time.time()-start_time)

            residual_sln = jnp.zeros(self.grid[0].shape)

            if self.residuals_info == True:
                res = residual_func(opt_sln, z, f)
                residual_sln = jnp.mean(jnp.abs(res))
                print('mean abs Residual: %g' % residual_sln  )
                print('norm error sqrd', jnp.linalg.norm(res, axis=-1)**2.)
            
            else:
                return opt_sln.reshape(self.grid[0].shape)

        opt_sln, res = opt_sln.reshape(self.grid[0].shape), res.reshape(self.grid[0].shape)

        return opt_sln, res, residual_sln


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import time
    import os
    # os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import jax
    import time
    direc = 'tmp-Poisson2DFEMRes'
    os.makedirs(direc, exist_ok=True)

    nx = 10
    solver = Poisson2DSolver(nx, verbose=True)
    # solver.residual_evaluator = solver.residual_WeakForm_Hat
    # solver.residual_evaluator = solver.residual_WeakForm_Hat_shift_indexing
    print(solver.grid[0], '\n', solver.grid[1])
    u = jnp.sin(jnp.pi*solver.grid[0])*jnp.sin(jnp.pi*solver.grid[1]**0.)
    z = u ** 0.
    f = u ** 0. * 1.
    f = solver.grid[0]**2.*solver.grid[1]**2.
    # f = jnp.sin(3*jnp.pi*solver.grid[0])*jnp.sin(3*jnp.pi*solver.grid[1])

    # start_time = time.time()
    # res_jit = jax.jit(solver.residual)
    # res_bind = lambda u : res_jit(u, z.reshape(-1,), f.reshape(-1,), solver.grid.reshape(-1,))
    # u_sln = jaxopt.linear_solve.solve_gmres(res_bind, jnp.zeros(u.reshape(-1,).shape))
    # # u_sln = jaxopt.linear_solve.solve_cg(res_bind, jnp.zeros(u.reshape(-1,).shape), tol=1e-9)
    # # u_sln = jaxopt.linear_solve.solve_bicgstab(res_bind, jnp.zeros(u.reshape(-1,).shape))
    # # u_sln, info = jax.scipy.sparse.linalg.cg(res_bind, jnp.zeros(u.reshape(-1,).shape), x0=None, tol=1e-05, atol=0.0, maxiter=None, M=None)
    
    # print(f'time taken lin_solve = {time.time() - start_time:.3f}s')
    # plt.contourf(solver.grid[0], solver.grid[1], jnp.where(jnp.abs(u_sln)<1e-8, 0, u_sln).reshape(solver.grid[0].shape), 50)
    # plt.colorbar()
    # plt.savefig(direc + '/2DPoisson_test_linsolver.png')
    # plt.close()

    # start_time = time.time()
    # u_sln, res, res_sln = solver.solve(u.reshape(-1,), z.reshape(-1,), f.reshape(-1,), solver.grid.reshape(-1,))
    # print(f'time taken = {time.time() - start_time:.3f}s')
    # plt.contourf(solver.grid[0], solver.grid[1], jnp.where(jnp.abs(u_sln)<1e-8, 0, u_sln), 50)
    # plt.colorbar()
    # plt.savefig(direc + '/2DPoisson_test.png')
    # plt.close()
    # print(u_sln[0,0], u_sln[-1,0], u_sln[0,-1], u_sln[-1,-1])
    # print(u_sln[0,0], u_sln[1,0], u_sln[0,1])
    # # print(u_sln)


    # u_mms = jnp.sin(1*jnp.pi*solver.grid[0])*jnp.sin(1*jnp.pi*solver.grid[1])

    # plt.contourf(solver.grid[0], solver.grid[1], jnp.where(jnp.abs(u_mms)<1e-8, 0, u_mms), 50)
    # plt.colorbar()
    # plt.savefig(direc + '/2DPoisson_u_mms.png')
    # plt.close()
    # f = 2*jnp.pi**2 *  jnp.sin(1*jnp.pi*solver.grid[0]) * jnp.sin(1*jnp.pi*solver.grid[1])
    # # u_sln, res, res_sln = solver.solve(u.reshape(-1,), z.reshape(-1,), f.reshape(-1,), solver.grid.reshape(-1,))
    # res_bind = lambda u : res_jit(u, z.reshape(-1,), f.reshape(-1,), solver.grid.reshape(-1,))
    # start_time = time.time()
    # u_sln = jaxopt.linear_solve.solve_gmres(res_bind, jnp.zeros(u.reshape(-1,).shape))
    # u_sln = u_sln.reshape(solver.grid[0].shape)
    # print(f'time taken lin_solve gmres= {time.time() - start_time:.3f}s')
    # print('error mms', jnp.sqrt(jnp.sum((u_sln - u_mms)**2. * solver.dx)))
    # plt.contourf(solver.grid[0], solver.grid[1], jnp.where(jnp.abs(u_sln)<1e-8, 0, u_sln), 50)
    # plt.colorbar()
    # plt.savefig(direc + '/2DPoisson_u_mms_sln.png')
    # plt.close()

    # res_jit = jax.jit(solver.residual)
    # res = res_jit(u.reshape(-1,), z.reshape(-1,), f.reshape(-1,), solver.grid.reshape(-1,))
    # start_time = time.time()
    # n = 100
    # for i in range(n):
    #     res = res_jit(u.reshape(-1,), z.reshape(-1,), f.reshape(-1,), solver.grid.reshape(-1,))
    # print(f'res_jit_time:{(time.time()-start_time)/n}')
    u = jnp.sin(jnp.pi*solver.grid[0])*jnp.sin(jnp.pi*solver.grid[1])
    res_bind = lambda z: solver.residual(u, z, f, with_bc = False, rhs_lhs = None)
    u = u.reshape(-1,)
    from jax import random
    key = random.PRNGKey(0)
    z   = jnp.sin(solver.grid[0]*jnp.pi) * jnp.sin(solver.grid[1]*jnp.pi) + 0.1
    r, Jrowsum_z = jax.jvp(res_bind, (z,), (jnp.ones(z.shape),))


    u_mms = jnp.sin(jnp.pi*solver.grid[0])*jnp.sin(jnp.pi*solver.grid[1])
    plt.contourf(solver.grid[0], solver.grid[1], jnp.where(jnp.abs(u_mms)<1e-8, 0, u_mms), 50)
    plt.colorbar()
    plt.savefig(direc + '/2DPoisson_u_mms_2.png')
    plt.close()
    z = 1. + solver.grid[0] + solver.grid[1]
    plt.contourf(solver.grid[0], solver.grid[1], z, 50)
    plt.colorbar()
    plt.savefig(direc + '/2DPoisson_u_mms_z_2.png')
    plt.close()
    f = 2 * ( 1. + solver.grid[0] + solver.grid[1] ) * jnp.pi**2. * jnp.sin(jnp.pi*solver.grid[0]) * jnp.sin(jnp.pi*solver.grid[1]) \
        - jnp.pi * (jnp.cos(jnp.pi*solver.grid[0])*jnp.sin(jnp.pi*solver.grid[1])+jnp.sin(jnp.pi*solver.grid[0])*jnp.cos(jnp.pi*solver.grid[1]))
    plt.contourf(solver.grid[0], solver.grid[1], f, 50)
    plt.colorbar()
    plt.savefig(direc + '/2DPoisson_u_mms_f_2.png')
    plt.close()
    solver.solverChoice = 'cg' # 'cg' || 'gmres' || 'bicgstab' 
    solver.residuals_info = True
    u_sln, res, res_sln = solver.solve(u.reshape(-1,), z.reshape(-1,), f.reshape(-1,))
    print('error mms 2', jnp.sqrt(jnp.sum((u_sln - u_mms)**2. * solver.dx**2.)))
    print('error mms 2', jnp.linalg.norm(u_sln - u_mms)/jnp.linalg.norm(u_mms))
    plt.contourf(solver.grid[0], solver.grid[1], jnp.where(jnp.abs(u_sln)<1e-8, 0, u_sln), 50)
    plt.colorbar()
    plt.savefig(direc + '/2DPoisson_u_mms_2_sln.png')
    plt.close()