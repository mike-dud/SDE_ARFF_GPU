import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

from jax import jit, vmap
from functools import partial

import matplotlib.pyplot as plt
import time

#from numpy import linalg as LA
from matplotlib.colors import Normalize
from multiprocessing import Pool
import scipy




class NNHyperparameters:
    def __init__(self, K=2**6, M_min=0, M_max=100, lambda_reg=2e-3, gamma=1, delta=0.1, name=None):
        self.K = K
        self.M_min = M_min
        self.M_max = M_max
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.delta = delta
        self.name = name


class SDEARFFTrain:
    def __init__(self, n_dimensions=1, x_min=None, x_max=None, omega_drift=None, amp_drift=None, z_mean=0, z_std=1, omega_diffusion=None, amp_diffusion=None, diffusion_std=1, diffusion_type="diagonal", random_seed=None, resampling=True, metropolis_test=True):
        self.d = n_dimensions
        self.tri = n_dimensions * (n_dimensions + 1) // 2
        self.x_min = x_min
        self.x_max = x_max
        self.omega_drift = omega_drift
        self.amp_drift = amp_drift
        self.z_mean = z_mean
        self.z_std = z_std
        self.omega_diffusion = omega_diffusion
        self.amp_diffusion = amp_diffusion
        self.diffusion_std = diffusion_std
        self.diffusion_type = diffusion_type
        self.key = random.PRNGKey(random_seed)
        self.resampling = resampling
        self.metropolis_test = metropolis_test
        self.history = {'loss': None, 'val_loss': None, 'true_loss': None, 'training_time': None, 'drift_error': None, 'diffusion_error': None}

    @staticmethod
    def normalise_z(z):
        z_mean = jnp.mean(z, axis=0)
        z_std = jnp.std(z, axis=0)
        z_norm = (z - z_mean) / z_std
        return z_norm, z_mean, z_std

    @staticmethod
    def normalise_diffusion_vectors(diffusion_vectors):
        diffusion_std = jnp.mean(diffusion_vectors, axis=0)
        diffusion_vectors_norm = diffusion_vectors / diffusion_std
        return diffusion_vectors_norm, diffusion_std

    def split_data(self, key, validation_split, *inputs):
        num_samples = inputs[0].shape[0]
        valid_sample_size = int(num_samples * validation_split)

        # Split the RNG to get a fresh subkey (pure functional RNG)
        key, subkey = random.split(key)
    
        # Get a random permutation of indices
        permuted_indices = random.permutation(subkey, num_samples)
    
        # Take the first `valid_sample_size` indices as validation
        valid_indices = permuted_indices[:valid_sample_size]
    
        # Create a boolean mask: True for training, False for validation
        mask = jnp.ones(num_samples, dtype=bool).at[valid_indices].set(False)
    
        # Split the data
        inputs_train = tuple(data[mask] for data in inputs)
        inputs_valid = tuple(data[~mask] for data in inputs)

        return inputs_train, inputs_valid, key

    @staticmethod
    @jit
    def S(x, omega):
        x_omega = jnp.matmul(x, omega)
        S_ = jnp.exp(1j * x_omega)
        return S_

    @staticmethod
    @jit
    def beta(x, omega, amp):
        beta_ = jnp.real(jnp.matmul(SDEARFFTrain.S(x, omega), amp))
        return beta_

    @staticmethod
    def matrix_sqrtm(matrix):
        return scipy.linalg.sqrtm(matrix)

    @staticmethod
    def matrix_cholesky(matrix):
        return scipy.linalg.cholesky(matrix)

    def covariance(self, x):
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        covariance_vectors = SDEARFFTrain.beta(x_norm, self.omega_diffusion, self.amp_diffusion) * self.diffusion_std
        
        if self.diffusion_type == "diagonal":
            covariance_matrix = jax.vmap(jnp.diag)(covariance_vectors)
        else:
            LT_idx = jnp.tril_indices(self.d)
            LT = jnp.zeros((x.shape[0], self.d, self.d))
            LT = LT.at[:, LT_idx[0], LT_idx[1]].set(covariance_vectors[:, :self.tri])
            UT = jnp.swapaxes(LT, 1, 2)
            diag = jnp.diagonal(LT, axis1=1, axis2=2)
            covariance_matrix = LT + UT - jnp.eye(self.d) * diag[:, None, :]

        return covariance_matrix

    def diffusion(self, x):
        if x.ndim == 1:
            x = x[None, :] 
        
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        covariance = SDEARFFTrain.covariance(self, x)
        
        if self.diffusion_type == "diagonal":
            diffusion_matrix = jnp.sqrt(covariance)
        elif self.diffusion_type == "triangular":
            diffusion_matrix = jax.vmap(jnp.linalg.cholesky)(covariance)
        else:
            def matrix_sqrtm(mat):
                vals, vecs = jnp.linalg.eigh(mat)
                sqrt_vals = jnp.sqrt(jnp.clip(vals, a_min=0.0))
                return (vecs * sqrt_vals) @ vecs.T
            diffusion_matrix = jax.vmap(matrix_sqrtm)(covariance)
                  
        return diffusion_matrix

    def drift(self, x):
        if x.ndim == 1:
            x = x[None, :]
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        drift_ = (SDEARFFTrain.beta(x_norm, self.omega_drift, self.amp_drift) * self.z_std + self.z_mean)
        return drift_

    def drift_diffusion(self, x):
        return SDEARFFTrain.drift(self, x), SDEARFFTrain.diffusion(self, x)

    def get_diffusion_vectors(self, y_n, y_np1, x, step_sizes):
        f = y_np1 - (y_n + step_sizes * SDEARFFTrain.drift(self, x))
        if self.diffusion_type == "diagonal":
            diffusion_vectors = f ** 2 / step_sizes
        else:
            f_reshaped = f[:, :, None]
            f_square = jnp.matmul(f_reshaped, f_reshaped.transpose(0, 2, 1))
            f_square_h = f_square / step_sizes[:, None]

            LT_idx_i, LT_idx_j = jnp.tril_indices(self.d)
            diffusion_vectors = f_square_h[:, LT_idx_i, LT_idx_j]

        return diffusion_vectors
    
    @staticmethod
    @jit
    def get_amp(x, y, lambda_reg, omega):
        S = SDEARFFTrain.S(x, omega) 

        A = jnp.matmul(jnp.conj(jnp.transpose(S)), S) + x.shape[0] * lambda_reg * jnp.eye(omega.shape[1])
        b = jnp.matmul(jnp.conj(jnp.transpose(S)), y)
        
        #amp = jnp.linalg.solve(A, b)
        amp, _ = jax.scipy.sparse.linalg.cg(A, b, tol=1e-6, maxiter=1e5)
        return amp

    @staticmethod
    def batch_logpdf_diag(x, mean, scale_diag):
        diff = x - mean
        var = scale_diag ** 2
        log_det = jnp.sum(jnp.log(var), axis=-1)
        maha = jnp.sum((diff ** 2) / var, axis=-1)
        d = x.shape[-1]
        log_prob = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det + maha)
        #jax.debug.print("diff: {0}, var: {1}, log_det: {2}, maha: {3}, log_prob: {4}", diff, var, log_det, maha, log_prob)

        return log_prob

    @staticmethod
    def batch_logpdf_full(x, mean, chol):
        diff = x - mean
        solve = jax.scipy.linalg.cho_solve((chol, True), diff)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
        maha = jnp.sum(diff * solve)
        d = x.shape[-1]
        log_prob = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det + maha)
        return log_prob

    @staticmethod
    @jit
    def batched_logpdf_diag(x, mean, scale_diag):
        return jax.vmap(SDEARFFTrain.batch_logpdf_diag, in_axes=(0, 0, 0))(x, mean, scale_diag)

    @staticmethod
    @jit
    def batched_logpdf_full(x, mean, chol):
        return jax.vmap(SDEARFFTrain.batch_logpdf_full, in_axes=(0, 0, 0))(x, mean, chol)
    
    @staticmethod
    def chunked_apply(fn, y, loc, scale, chunk_size=4096):
        N = y.shape[0]
        results = []
        for i in range(0, N, chunk_size):
            y_chunk = y[i:i+chunk_size]
            loc_chunk = loc[i:i+chunk_size]
            scale_chunk = scale[i:i+chunk_size]
            results.append(fn(y_chunk, loc_chunk, scale_chunk))
        return jnp.concatenate(results, axis=0)
    
    @staticmethod
    def get_loss(y_n, y_np1, x_n, step_sizes, drift, diffusion, diffusion_type="symmetric"):
        loc = y_n + step_sizes * drift(x_n)
        scale = jnp.sqrt(step_sizes[:, 0])[:, None, None] * diffusion(x_n)
        
        if diffusion_type == "symmetric":
            scale = jnp.matmul(scale, jnp.transpose(scale, (0, 2, 1)))
            scale = jnp.linalg.cholesky(scale)

        if diffusion_type == "diagonal":
            scale = jnp.diagonal(scale, axis1=1, axis2=2)
            log_prob = SDEARFFTrain.chunked_apply(SDEARFFTrain.batched_logpdf_diag, y_np1, loc, scale)
        else:
            log_prob = SDEARFFTrain.chunked_apply(SDEARFFTrain.batched_logpdf_full, y_np1, loc, scale)

        return -jnp.mean(log_prob)

    @staticmethod
    def RMSE(trained_func, true_func, x):
        diff = trained_func(x).reshape(-1, 1, 1) - true_func(x).reshape(-1, 1, 1)
        rmse = jnp.sqrt(jnp.mean(diff ** 2))
        return rmse

    @staticmethod
    @jit
    def vnorm(x):
        return jax.vmap(jnp.linalg.norm, in_axes=(0,))(x)
    
    @staticmethod
    @partial(jit, static_argnames=['RESAMPLING', 'METROPOLIS_TEST'])
    def ARFF_one_step(key, omega, amp, x, y, delta, lambda_reg, gamma, RESAMPLING=True, METROPOLIS_TEST=True):
        amp_norm = SDEARFFTrain.vnorm(amp)
        
        if RESAMPLING:
            amp_pmf = amp_norm / jnp.sum(amp_norm)
            key, subkey = random.split(key)
            omega = omega[:, random.choice(subkey, omega.shape[1], shape=(omega.shape[1],), p=amp_pmf)]
            
        if METROPOLIS_TEST:
            key, subkey = random.split(key)
            dw = random.normal(subkey, omega.shape)
            omega_prime = omega + delta * dw
    
            amp_prime_norm = SDEARFFTrain.vnorm(SDEARFFTrain.get_amp(x, y, lambda_reg, omega_prime))
    
            key, subkey = random.split(key)
            omega = jnp.where((amp_prime_norm / amp_norm) ** gamma >= random.uniform(subkey, omega.shape[1]), omega_prime, omega)
               
        else:
            key, subkey = random.split(key)
            dw = random.normal(subkey, omega.shape)
            omega = omega + delta * dw

        amp = SDEARFFTrain.get_amp(x, y, lambda_reg, omega)
    
        return omega, amp, key
        
    def ARFF_train(self, key, param, x, y_norm, validation_split):
        start_time = time.time()
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        (x_norm, y_norm), (x_norm_valid, y_norm_valid), key = SDEARFFTrain.split_data(self, key, validation_split, x_norm, y_norm)

        omega = jnp.zeros((x.shape[1], param.K))
        amp = SDEARFFTrain.get_amp(x_norm, y_norm, param.lambda_reg, omega)

        ve = jnp.zeros(param.M_max)
        ve_min = jnp.inf
        moving_avg = jnp.zeros(param.M_max)
        min_moving_avg = jnp.inf
        moving_avg_len = 5
        min_index = 0
        break_iterations = 5
        
        for i in range(param.M_max):
            # ARFF one step
            omega, amp, key = SDEARFFTrain.ARFF_one_step(key, omega, amp, x_norm, y_norm, 
                                                         param.delta, param.lambda_reg, param.gamma, 
                                                         RESAMPLING=self.resampling,
                                                         METROPOLIS_TEST=self.metropolis_test)

            ve = ve.at[i].set(jnp.mean(jnp.abs(SDEARFFTrain.beta(x_norm_valid, omega, amp) - y_norm_valid) ** 2))
            moving_avg = moving_avg.at[i].set(jnp.where(i < moving_avg_len,
                                      jnp.mean(ve[:i+1]),
                                      jnp.mean(ve[i-moving_avg_len+1:i+1])))
        
            if moving_avg[i] < min_moving_avg:
                min_moving_avg = moving_avg[i]
                min_index = i
        
            if min_index + break_iterations < i and i > param.M_min:
                break
        
            if ve[i] < ve_min:
                end_time = time.time()
                ve_min = ve[i]
                setattr(self, f'omega_{param.name}', omega)
                setattr(self, f'amp_{param.name}', amp)
        
            print(f"\r{param.name} epoch: {i}", end='')
        print()

        return ve[:i], moving_avg[:i], end_time-start_time, key

    def train_model(self, drift_param, diffusion_param, true_drift, true_diffusion, y_n, y_np1, x=None, step_sizes=None, validation_split=0.1, ARFF_validation_split=0.1, YinX=True, plot=False):

        key = self.key
        
        y_n = jnp.array(y_n)
        y_np1 = jnp.array(y_np1)
        step_sizes = jnp.array(step_sizes)
        
        if x is None:
            x = y_n
        elif YinX:
            x = jnp.concatenate((y_n, x), axis=1)
        else:
            x = jnp.array(x)

        (y_n, y_np1, x, step_sizes), (y_n_valid, y_np1_valid, x_valid, step_sizes_valid), key = SDEARFFTrain.split_data(self, key, validation_split, y_n, y_np1, x, step_sizes)

        self.x_min = jnp.min(x, axis=0)
        self.x_max = jnp.max(x, axis=0)

        # calculate z
        z_start = time.time()
        z = (y_np1 - y_n)/step_sizes
        z_norm, self.z_mean, self.z_std = SDEARFFTrain.normalise_z(z)
        z_time = time.time() - z_start

        # train for z using ARFF
        ve_drift, moving_avg_drift, minima_time_drift, key = SDEARFFTrain.ARFF_train(self, key, drift_param, x, z_norm, ARFF_validation_split)

        if plot:
            if x.shape[1] == 1:
                SDEARFFTrain.plot_1D(self, true_drift, x, z_norm, drift_param.name)
            elif x.shape[1] == 2:
                SDEARFFTrain.plot_2D(self, true_drift, x, z_norm, drift_param.name)
            SDEARFFTrain.plot_loss(ve_drift, moving_avg_drift)
        
        # calculate point-wise diffusion
        diffusion_vector_start = time.time()
        diffusion_vectors = SDEARFFTrain.get_diffusion_vectors(self, y_n, y_np1, x, step_sizes)
        diffusion_vectors_norm, self.diffusion_std = SDEARFFTrain.normalise_diffusion_vectors(diffusion_vectors)
        diffusion_vector_time = time.time() - diffusion_vector_start

        # train for global diffusion using ARFF
        ve_diffusion, moving_avg_diffusion, minima_time_diffusion, key = SDEARFFTrain.ARFF_train(self, key, diffusion_param, x, diffusion_vectors_norm, ARFF_validation_split)
        
        if plot:
            if x.shape[1] == 1:
                SDEARFFTrain.plot_1D(self, true_diffusion, x, diffusion_vectors_norm, diffusion_param.name)
            elif x.shape[1] == 2:
                SDEARFFTrain.plot_2D(self, true_diffusion, x, diffusion_vectors_norm, diffusion_param.name)
            SDEARFFTrain.plot_loss(ve_diffusion, moving_avg_diffusion)

        # calculate losses
        self.history['drift_RMSE'] = SDEARFFTrain.RMSE(self.drift, true_drift, x)
        self.history['diffusion_RMSE'] = SDEARFFTrain.RMSE(self.diffusion, true_diffusion, x)
        self.history['loss'] = SDEARFFTrain.get_loss(y_n, y_np1, x, step_sizes, drift=self.drift, diffusion=self.diffusion, diffusion_type=self.diffusion_type)
        self.history['val_loss'] = SDEARFFTrain.get_loss(y_n_valid, y_np1_valid, x_valid, step_sizes_valid, drift=self.drift, diffusion=self.diffusion, diffusion_type=self.diffusion_type)
        self.history['true_loss'] = SDEARFFTrain.get_loss(y_n_valid, y_np1_valid, x_valid, step_sizes_valid, drift=true_drift, diffusion=true_diffusion, diffusion_type=self.diffusion_type)
        self.history['training_time'] = z_time + diffusion_vector_time + minima_time_drift + minima_time_diffusion
        
        print(f"\rDrift RMSE: {self.history['drift_RMSE']}")
        print(f"\rDiffusion RMSE: {self.history['diffusion_RMSE']}")
        print(f"\rObserved loss: {self.history['loss']}")
        print(f"\rObserved validation loss: {self.history['val_loss']}")
        print(f"\rTrue loss: {self.history['true_loss']}")
        print(f"\rTraining time: {self.history['training_time']}")
        return self
        
    # plot functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_1D(self, true_func, x, y_norm, name):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})
    
        # grid
        x_div = 500
        x_grid = np.linspace(self.x_min, self.x_max, x_div).reshape((x_div, 1))
        x_grid_norm = np.linspace(0, 1, x_div).reshape((x_div, 1))

        # plot training data
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        ax[0].scatter(x_norm[:, 0], y_norm, alpha=0.5)

        # plot intermediate function
        omega = getattr(self, f"omega_{name}")
        amp = getattr(self, f"amp_{name}")
        ax[1].plot(x_grid_norm, SDEARFFTrain.beta(x_grid_norm, omega, amp))

        # plot trained drift/diffusion
        func = getattr(SDEARFFTrain, name)
        ax[2].plot(x_grid, func(self, x_grid).reshape((x_div, 1)), label="Trained")
        
        # plot actual drift/diffusion
        ax[2].plot(x_grid, true_func(x_grid).reshape((x_div, 1)), label="True")

        # set labels
        ax[0].set_ylabel(f'{name}', fontsize=12)

        ax[0].set_xlabel(r'$\bar{x}_0$', fontsize=12)
        ax[1].set_xlabel(r'$\bar{x}_0$', fontsize=12)
        ax[2].set_xlabel(r'$x_0$', fontsize=12)

        ax[0].set_title('Training Data', fontsize=12)
        ax[1].set_title('Intermediate', fontsize=12)
        ax[2].set_title('Trained and True', fontsize=12)

        ax[2].legend()

        plt.show()

    def plot_2D(self, true_func, x, y_norm, name):
        output_dim = y_norm.shape[1]
        
        # grid
        x_div = 30
        y_div = 30
        x_1, x_2 = np.meshgrid(np.linspace(self.x_min[0], self.x_max[0], x_div), np.linspace(self.x_min[1], self.x_max[1], y_div))
        x_1_norm, x_2_norm = np.meshgrid(np.linspace(0, 1, x_div), np.linspace(0, 1, y_div))
        x_grid = np.column_stack((x_1.ravel(), x_2.ravel()))
        x_norm_grid = np.column_stack((x_1_norm.ravel(), x_2_norm.ravel()))
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)

        # get true and trained grid results
        omega = getattr(self, f"omega_{name}")
        amp = getattr(self, f"amp_{name}")
        func = getattr(SDEARFFTrain, name)

        intermediate = SDEARFFTrain.beta(x_norm_grid, omega, amp)
        trained = func(self, x_grid)
        true_ = true_func(x_grid)

        # determine No. plots required
        if name == "diffusion" and output_dim != 1:
            if self.diffusion_type == "diagonal":
                trained = trained[:, [0, 1], [0, 1]]
                true_ = true_[:, [0, 1], [0, 1]]
            else:
                trained = trained[:, [0, 0, 1], [0, 1, 1]]
                true_ = true_[:, [0, 0, 1], [0, 1, 1]]
        elif output_dim == 1:
            true_ = true_.reshape(-1, 1)

        fig, ax = plt.subplots(output_dim, 4, figsize=(20, 4*output_dim), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})

        if output_dim == 1:
            ax = np.expand_dims(ax, axis=0)
        
        for j in range(output_dim):
            # get norms 
            norms_1_2 = Normalize(vmin=min(np.real(y_norm[:, j]).min(), np.real(intermediate[:, j]).min()), 
                      vmax=max(np.real(y_norm[:, j]).max(), np.real(intermediate[:, j]).max()))
            norms_3_4 = Normalize(vmin=min(np.real(trained[:, j]).min(), np.real(true_[:, j]).min()), 
                          vmax=max(np.real(trained[:, j]).max(), np.real(true_[:, j]).max()))

            # plot training data
            ax[j, 0].scatter(x_norm[:, 0], x_norm[:, 1], c=y_norm[:, j].real, cmap='viridis', s=20, norm=norms_1_2)

            # plot intermediate function
            int = ax[j, 1].imshow(np.real(np.reshape(intermediate[:, j], (x_div, -1))), cmap='viridis', extent=[0, 1, 0, 1], origin='lower', aspect='auto', norm=norms_1_2)
        
            # plot trained drift/diffusion
            tr = ax[j, 2].imshow(np.real(np.reshape(trained[:, j], (x_div, -1))), cmap='viridis', extent=[self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1]], origin='lower', aspect='auto', norm=norms_3_4)

            # plot actual drift/diffusion
            ax[j, 3].imshow(np.reshape(true_[:, j], (x_div, -1)), cmap='viridis', extent=[self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1]], origin='lower', aspect='auto', norm=norms_3_4)

            # add color bars
            cbar_1_2 = fig.colorbar(int, ax=ax[j, 0], orientation='vertical', fraction=0.02, pad=0.04)
            #cbar_1_2.set_label(f'Row {j+1} Color Scale (1 & 2)')

            cbar_3_4 = fig.colorbar(tr, ax=ax[j, 2], orientation='vertical', fraction=0.02, pad=0.04)
            #cbar_3_4.set_label(f'Row {j+1} Color Scale (3 & 4)')

            ax[j, 0].set_ylabel(fr'${name}_{{{j}}}(x_0)$', fontsize=12)

        # format
        for axes in ax.flatten():
            axes.tick_params(axis='both', which='both', labelsize=6)
        plt.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.03)

        # set labels
        ax[0, 0].set_title('Training Data', fontsize=12)
        ax[0, 1].set_title('Intermediate', fontsize=12)
        ax[0, 2].set_title('Trained', fontsize=12)
        ax[0, 3].set_title('True', fontsize=12)

        plt.show()
    
    @staticmethod
    def plot_loss(ve, moving_avg):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})

        ax.semilogy(ve, label="Validation Error")
        ax.semilogy(moving_avg, label="Moving Average")

        ax.set_title('ARFF Loss', fontsize=12)
        ax.set_xlabel(r'$M$', fontsize=12)
        ax.legend()

        plt.show()

# # batched
# SDEARFFTrain.batched_logpdf_diag = jax.vmap(SDEARFFTrain.batch_logpdf_diag, in_axes=(0, 0, 0))
# SDEARFFTrain.batched_logpdf_full = jax.vmap(SDEARFFTrain.batch_logpdf_full, in_axes=(0, 0, 0))


