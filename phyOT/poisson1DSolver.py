import os
import jax
import jax.numpy as jnp
import jaxopt
import sys
from functools import partial

from phyOT.utils import *

class PoissonSolver():
    '''
        class with methods for solving 1D Poisson from d/dx( \theta * du/dx) - f = 0
    '''
    def __init__(self, nx, verbose):

        self.verbose = verbose
        self.nx = nx
        self.grid = self.make_grid(self.nx)
        self.dx = self.grid[1] - self.grid[0]
        self.sln_shape = self.grid.shape
        self.u_init = self.grid*0.
        self.solverChoice = 'root' #! 'lm' || 'root'
        self.residual = self.residual_WeakForm_Hat
        self.residuals_info = True
        self.device = 'cuda'

    def make_grid(self, nx):
        X = jnp.linspace(0., 1., nx)
        return X
    
    def inter1d_wrapper(self, u, x):
        '''
        u as [nx] grid
        x as [nx] grid
        '''
        return jnp.interp(x, self.grid, u)

    def boundary_func(self, u, grid):
        return u * jnp.sin(jnp.pi * grid)

    def residual_WeakForm_Hat(self, u, z, f, with_bc = True, rhs_lhs = None):
        
        h = self.grid[1] - self.grid[0]

        z_l = jnp.pad(z[:-1], (1,0), constant_values=0.)

        u_l = jnp.pad(u[:-1], (1,0), constant_values=0.)
        u_r = jnp.pad(u[1:],  (0,1) , constant_values=0.)

        vx_ux_l = z_l *  (u - u_l)/h
        vx_ux_l = vx_ux_l.at[0].set(0.)
        vx_ux_r = z *  (u_r - u)/h
        vx_ux_r = vx_ux_r.at[-1].set(0.)

        fv_l = ( jnp.pad(f[:-1], [(1,0)]) ) * h / 2.
        fv_r = ( jnp.pad(f[1:],  [(0,1)]) ) * h / 2.
        fv   = fv_l + fv_r

        r = - (vx_ux_l - vx_ux_r) + fv

        # print('z', z)
        # print('vx_ux_l - vx_ux_r', vx_ux_l - vx_ux_r)
        # print('vx_ux_l',vx_ux_l)
        # print('vx_ux_r', vx_ux_r)
        # print('fv', fv)

        if with_bc == False:
            return r[1:-1]
        
        else:
            # # boundary residual
            res_l   = 0. - u[ :1]
            res_r   = 0. - u[-1:]

            r = jnp.concatenate((res_l, r[1:-1], res_r ), axis=0)

            return r


    def residual_Au(self, u, z):
        u  = jnp.pad(u, [1,1])
        h = self.grid[1] - self.grid[0]

        z_l = jnp.pad(z[:-1], (1,0), constant_values=0.)

        u_l = jnp.pad(u[:-1], (1,0), constant_values=0.)
        u_r = jnp.pad(u[1:],  (0,1) , constant_values=0.)

        vx_ux_l = z_l *  (u - u_l)/h
        vx_ux_l = vx_ux_l.at[0].set(0.)
        vx_ux_r = z *  (u_r - u)/h
        vx_ux_r = vx_ux_r.at[-1].set(0.)

        r = - (vx_ux_l - vx_ux_r)
        r = r[1:-1]

        return r
    


    
    def solve(self, u_init, z, f):

        with jax.default_device(jax.devices(self.device)[0]):

            if self.solverChoice == 'root':

                opt = jaxopt.ScipyRootFinding(
                        optimality_fun = self.residual,
                        method='hybr',
                        tol = 1e-6,
                        jit=True, implicit_diff_solve=jaxopt.linear_solve.solve_cg, 
                        has_aux=False,
                        options={'disp':True, 'xtol':1e-6, 'maxfev':200}, use_jacrev=False)
                
            elif self.solverChoice == 'lm':
                opt = jaxopt.LevenbergMarquardt(self.residual, maxiter=100, damping_parameter=1e-06,
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
                print('L2 opt error', opt.l2_optimality_error(opt_sln, self.grid, z))

            elif self.solverChoice == 'linear':
                print('USING CG')
                h = self.grid[1] - self.grid[0]
                fv_l = ( jnp.pad(f[:-1], [(1,0)]) ) * h / 2.
                fv_r = ( jnp.pad(f[1:],  [(0,1)]) ) * h / 2.
                fv   = fv_l + fv_r
                res_bind = lambda u : self.residual_Au(u, z.reshape(-1,))
                res_jit = jax.jit(res_bind)
                opt_sln = jaxopt.linear_solve.solve_cg(res_jit, -fv[1:-1])
                opt_sln = jnp.pad(opt_sln, [1,1])
            residual_sln = jnp.zeros((1,))

            if self.residuals_info == True:
                res = self.residual(opt_sln, z, f)
                residual_sln = jnp.mean(jnp.abs(res))
                print('mean abs Residual: ', residual_sln  )
                print('norm error sqrd', jnp.linalg.norm(res, axis=-1)**2.)
            else:
                return opt_sln.reshape(self.grid.shape)
            
        opt_sln = opt_sln.at[0].set(0.)
        opt_sln = opt_sln.at[-1].set(0.)

        return opt_sln, res, residual_sln


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import time
    import os
    # os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import jax
    direc = 'tmp-Poisson1D'
    os.makedirs(direc, exist_ok=True)

    nx = 1000
    solver = PoissonSolver(nx=nx, verbose=True)
    solver.solverChoice = 'root'
    z = solver.grid ** 0.
    f = solver.grid ** 0.

    u_true = -(solver.grid)**2./2. + 0.5 * solver.grid

    print('### Weaf Form ###')
    solver.residual = solver.residual_WeakForm_Hat
    u_init = solver.grid ** 0.
    start_time = time.time()
    u_sln, res, residual_sln = solver.solve(u_init, z, f)
    print(f'time solve = {time.time() - start_time:.4f}s')
    dx = solver.grid[1]-solver.grid[0]
    print('L2 error = ', jnp.sqrt(jnp.sum((u_true - u_sln)**2. * dx)) )

    ### For linear solver ###
    start_time = time.time()
    h = solver.grid[1] - solver.grid[0]
    fv_l = ( jnp.pad(f[:-1], [(1,0)]) ) * h / 2.
    fv_r = ( jnp.pad(f[1:],  [(0,1)]) ) * h / 2.
    fv   = fv_l + fv_r
    res_bind = lambda u : solver.residual_Au(u, z.reshape(-1,))
    res_jit = jax.jit(res_bind)
    opt_sln = jaxopt.linear_solve.solve_cg(res_jit, -fv[1:-1])
    print(f'time taken linear solver: {time.time()-start_time:.4f}')
    print('L2 error = ', jnp.sqrt(jnp.sum((u_true -  jnp.pad(opt_sln, [1,1]))**2. * dx)) )


    plt.plot(solver.grid, u_sln , label='sln - WF')
    plt.plot(solver.grid, u_true, '--', label='gt')
    plt.plot(solver.grid, jnp.pad(opt_sln, [1,1]), '--', label='CG')

    plt.legend()
    plt.grid()
    plt.title('test for Poisson root solver')
    plt.savefig(direc+'/solver-tmp2.pdf')
    plt.close()
    