import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
from phyOT.utils import activation_functions
from collections.abc import Callable
import os
import matplotlib.pyplot as plt

class FNO_utils1D():
    @staticmethod
    def RMult(R, f):
        # R   [modes, ch_in, ch_out]
        # f   [modes, ch_in]
        # out [modes, ch_out]
        f = f[:R.shape[0], ...]
        print('R.shape, f.shape:', R.shape, f.shape)
        return jnp.einsum('xio,xi->xo', R, f)
    @staticmethod
    def fftpad(v, Rf):
        return (v * 0.).at[:Rf.shape[0], :Rf.shape[1]].set(Rf)
    @staticmethod
    def get_conv(v):
        return nn.Conv(features = v.shape[-1], kernel_size=(1,), strides=(1,), padding='SAME')
    @staticmethod
    def get_shape_R(n_modes, v):
        return (n_modes, v.shape[-1], v.shape[-1])
    @staticmethod
    def get_fft_axes():
        return (0,)

class FNO_utils2D():
    @staticmethod
    def RMult(R, f):
        # R   [modes, modes, ch_in, ch_out]
        # f   [modes, modes, ch_in]
        # out [modes, modes, ch_out]
        f = f[ :R.shape[0], :R.shape[0], ...]
        print('R.shape, f.shape:', R.shape, f.shape)
        return jnp.einsum('xyio,xyi->xyo', R, f)
    @staticmethod
    def fftpad(v, Rf):
        return (v * 0.).at[:Rf.shape[0], :Rf.shape[1], :Rf.shape[2]].set(Rf)
    @staticmethod
    def get_conv(v):
        return nn.Conv(features = v.shape[-1], kernel_size=(1,1), strides=(1,1), padding='SAME')
    @staticmethod
    def get_shape_R(n_modes, v):
        return (n_modes, n_modes, v.shape[-1], v.shape[-1])
    @staticmethod
    def get_fft_axes():
        return (0,1)

class FLayer(nn.Module):
    n_modes: int
    FNO_utils: object #FNO_utils1D

    def complex_kernel_init(rng, shape):
        key1, key2 = random.split(rng)
        x = jax.lax.complex(random.uniform(key1, shape),  random.uniform(key2, shape))
        return x/(shape[-1]**2.)

    complex_kernel_init: Callable = complex_kernel_init

    @nn.compact
    def __call__(self, v):
        print('### START FNO Layer ###')
        print('v.shape: ', v.shape)

        W = self.FNO_utils.get_conv(v)(v)
        print('W.shape', W.shape)

        axes = self.FNO_utils.get_fft_axes( )
        f = jnp.fft.rfftn(v, axes=axes)
        print('f.shape:, f.type', f.shape)

        shape_R = self.FNO_utils.get_shape_R(self.n_modes, v)

        print('shape_R:', shape_R)
        R = self.param('R', self.complex_kernel_init, shape_R)

        Rf = self.FNO_utils.RMult(R, f)
        print('Rf:', Rf.shape)
        fp = self.FNO_utils.fftpad(f, Rf)
        print('fp.shape:', fp.shape)
        vi = jnp.fft.irfftn(fp, axes=axes)
        print('vi.shape: ', vi.shape)

        v_out = W + vi

        print('### END FNO Layer ###')
        return v_out


class FNO(nn.Module):
    cfg: dict
    FNO_utils: object

    '''
        cfg.dim_v
        cfg.n_modes
        cfg.out_dim
    '''

    @nn.compact
    def __call__(self, z):
        print('\n### START ###')

        v = z
        print('v.shape: ', v.shape)

        #! P layer
        v = nn.Dense(self.cfg.dim_v)(v)
        v = activation_functions[self.cfg.activation](v)
        v = nn.Dense(self.cfg.dim_v)(v)
        print('post p layer v.shape: ', v.shape)

        for _ in range(self.cfg.n_layers):
            v =  FLayer(self.cfg.n_modes, FNO_utils=self.FNO_utils)(v)
            v = activation_functions[self.cfg.activation](v)

        #! Q layer
        v = nn.Dense(self.cfg.dim_v)(v)
        v = activation_functions[self.cfg.activation](v)
        v = nn.Dense(self.cfg.out_dim)(v)
        print('post Q layer v.shape: ', v.shape)
        print('### END ###\n')
        return v

    def vmap_z_call(self, params, z):
        return jax.vmap(self.apply, in_axes=(None, 0))(params, z)

    def init_model(self, key, z):

        rng, key = random.split(key) # PRNG Key
        output, params = self.init_with_output(key, z)
        print('output.shape', output.shape)

        count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print('count', count)

        lr = optax.exponential_decay(
            self.cfg.learning_rate,
            self.cfg.n_decay_steps,
            self.cfg.decay_rate,
            staircase=True
        )
        if self.cfg.opt_type == 'adam':
            print('using ADAM')
            self.opt = optax.adam(learning_rate=lr)
        if self.cfg.opt_type == 'amsgrad':
            print('using AMSGRAD')
            self.opt = optax.amsgrad(learning_rate=lr)

        opt_state  = self.opt.init(params)

        print(self.tabulate(key, z, compute_flops=True, compute_vjp_flops=True))

        return params, opt_state

    def update(self, grads, params, opt_state):
        print('CONJUGATE UPDATE ONET') 
        grads = jax.tree_map(lambda x: x.conj(), grads)
        updates, opt_state = self.opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state


if __name__ == '__main__':

    x1 = jnp.linspace(0., 1., 500)
    x2 = x1
    grid = jnp.stack(jnp.meshgrid(x1,x2, indexing='ij'))

    z = jnp.sin(3*jnp.pi *grid[0]) * jnp.sin(3*jnp.pi*grid[1])

    from ml_collections import config_dict
    cfg = config_dict.ConfigDict()
    cfg.dim_v   = 32
    cfg.n_modes = 8
    cfg.out_dim = 1
    cfg.activation = 'silu'
    cfg.n_layers   = 4
    cfg.learning_rate = 1e-2
    cfg.n_decay_steps = int(1e3)
    cfg.decay_rate    = 0.9

    fno = FNO(cfg, FNO_utils2D)

    rng = random.key(0)
    key, rng = random.split(rng)

    zX = jnp.concatenate((z[None, ...], grid), 0)
    zX = jnp.transpose(zX, [1,2,0])
    params, opt_state = fno.init_model(key, zX)

    u = fno.vmap_z_call(params, zX[None, ...])[0]
    u = u[..., 0]

    os.makedirs('tmp_fno', exist_ok=True)
    plt.contourf(grid[0], grid[1], u, 50)
    plt.savefig('tmp_fno/'+'fno_test.png')
    plt.colorbar()
    plt.close

    def loss(params):
        u = fno.vmap_z_call(params, zX[None, ...])[0,:,:,0]
        u *= jnp.sin(jnp.pi *grid[0]) * jnp.sin(jnp.pi*grid[1])
        loss = jnp.linalg.norm(u - jnp.sin(9*jnp.pi *grid[0]) * jnp.sin(9*jnp.pi*grid[1]))**2.
        return loss

    @jax.jit
    def loss_and_update(params, opt_state):
        loss_and_grad = jax.value_and_grad(loss)
        loss_vals, grads = loss_and_grad(params)
        params, opt_state = fno.update(grads, params, opt_state)
        return loss_vals, params, opt_state
    
    LOSS = []
    for i in range(1_000):
        loss_vals, params, opt_state = loss_and_update(params, opt_state)
        LOSS.append(loss_vals)
        print(f'loss_val:{loss_vals:.2f}')
    
    plt.plot(LOSS)
    plt.savefig('tmp_fno/loss.pdf')
    plt.close()
    u = fno.vmap_z_call(params, zX[None, ...])[0,:,:,0]
    u *= jnp.sin(jnp.pi *grid[0]) * jnp.sin(jnp.pi*grid[1])
    plt.contourf(grid[0], grid[1], u, 50)
    plt.savefig('tmp_fno/fno_test_trained.png')
    plt.colorbar()
    plt.close
