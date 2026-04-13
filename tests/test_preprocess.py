import numpy as np
import cupy as cp
import pytest
from copy import deepcopy
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPU.MCMC_cu import _preprocess as _preprocess_gpu
from gstatsMCMC.MCMC import _preprocess as _preprocess_cpu

def to_numpy(x):
    """Convert CuPy array to NumPy if needed."""
    if hasattr(x, 'get'):
        return x.get()
    return np.asarray(x)

def make_test_inputs(shape=(20, 30), n_cond=15, seed=42):
    """Create shared test inputs for both functions."""
    rng = np.random.default_rng(seed)

    # Coordinate grids
    x = np.linspace(0, 100, shape[0])
    y = np.linspace(0, 100, shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Grid with NaN everywhere except conditioning points
    grid = np.full(shape, np.nan)
    cond_idx = rng.choice(shape[0] * shape[1], n_cond, replace=False)
    rows, cols = np.unravel_index(cond_idx, shape)
    grid[rows, cols] = rng.standard_normal(n_cond)

    variogram = {
        'major_range': 50.0,
        'minor_range': 25.0,
        'sill':        1.0,
        'nugget':      0.1,
        'vtype':       'spherical',
    }

    radius   = 30.0
    sim_mask = None   # test the "whole grid" branch
    stencil  = None   # test the stencil-creation branch

    return xx, yy, grid, variogram, sim_mask, radius, stencil



@pytest.fixture(scope='module')
def inputs():
    return make_test_inputs()


@pytest.fixture(scope='module')
def cpu_outputs(inputs):
    xx, yy, grid, variogram, sim_mask, radius, stencil = inputs
    return _preprocess_cpu(xx, yy, grid, variogram, sim_mask, radius, stencil)


@pytest.fixture(scope='module')
def gpu_outputs(inputs):
    xx, yy, grid, variogram, sim_mask, radius, stencil = inputs
    # GPU function expects CuPy arrays for grid / coordinate inputs
    return _preprocess_gpu(
        cp.asarray(xx), cp.asarray(yy), cp.asarray(grid),
        variogram, sim_mask, radius, stencil,
    )



class TestPreprocessOutputShape:
    def test_out_grid_shape(self, cpu_outputs, gpu_outputs, inputs):
        xx = inputs[0]
        cpu_grid = cpu_outputs[0]
        gpu_grid = to_numpy(gpu_outputs[0])
        assert cpu_grid.shape == gpu_grid.shape == xx.shape

    def test_cond_msk_shape(self, cpu_outputs, gpu_outputs, inputs):
        xx = inputs[0]
        assert cpu_outputs[1].shape == xx.shape
        assert to_numpy(gpu_outputs[1]).shape == xx.shape

    def test_inds_shape(self, cpu_outputs, gpu_outputs):
        cpu_inds = cpu_outputs[2]
        gpu_inds = to_numpy(gpu_outputs[2])
        assert cpu_inds.ndim == gpu_inds.ndim == 2
        assert cpu_inds.shape[1] == gpu_inds.shape[1] == 2
        assert cpu_inds.shape[0] == gpu_inds.shape[0], (
            f"Number of simulation points differs: CPU={cpu_inds.shape[0]}, GPU={gpu_inds.shape[0]}"
        )


class TestPreprocessOutputValues:
    ATOL = 1e-6

    def test_out_grid_values(self, cpu_outputs, gpu_outputs):
        cpu_grid = cpu_outputs[0]
        gpu_grid = to_numpy(gpu_outputs[0])
        # NaN positions must match
        np.testing.assert_array_equal(np.isnan(cpu_grid), np.isnan(gpu_grid))
        # Non-NaN values must match
        np.testing.assert_allclose(
            cpu_grid[~np.isnan(cpu_grid)],
            gpu_grid[~np.isnan(gpu_grid)],
            atol=self.ATOL,
        )

    def test_cond_msk_values(self, cpu_outputs, gpu_outputs):
        np.testing.assert_array_equal(
            cpu_outputs[1],
            to_numpy(gpu_outputs[1]),
        )

    def test_inds_values(self, cpu_outputs, gpu_outputs):
        cpu_inds = cpu_outputs[2]
        gpu_inds = to_numpy(gpu_outputs[2])
        # Sort both so order doesn't matter
        cpu_sorted = cpu_inds[np.lexsort(cpu_inds.T[::-1])]
        gpu_sorted = gpu_inds[np.lexsort(gpu_inds.T[::-1])]
        np.testing.assert_array_equal(cpu_sorted, gpu_sorted)

    def test_global_mean(self, cpu_outputs, gpu_outputs):
        np.testing.assert_allclose(
            cpu_outputs[4],
            float(to_numpy(gpu_outputs[4])),
            atol=self.ATOL,
        )

    def test_vario_keys_match(self, cpu_outputs, gpu_outputs):
        assert set(cpu_outputs[3].keys()) == set(gpu_outputs[3].keys())

    def test_vario_array_values(self, cpu_outputs, gpu_outputs, inputs):
        xx = inputs[0]
        cpu_vario = cpu_outputs[3]
        gpu_vario = gpu_outputs[3]
        for key in cpu_vario:
            cpu_val = cpu_vario[key]
            gpu_val = gpu_vario[key]
            if isinstance(cpu_val, np.ndarray):
                np.testing.assert_allclose(
                    cpu_val, to_numpy(gpu_val), atol=self.ATOL,
                    err_msg=f"Mismatch in variogram key '{key}'"
                )

    def test_stencil_shape(self, cpu_outputs, gpu_outputs):
        cpu_stencil = cpu_outputs[5]
        gpu_stencil = to_numpy(gpu_outputs[5])
        assert cpu_stencil.shape == gpu_stencil.shape, (
            f"Stencil shape mismatch: CPU={cpu_stencil.shape}, GPU={gpu_stencil.shape}"
        )

    def test_stencil_values(self, cpu_outputs, gpu_outputs):
        cpu_stencil = cpu_outputs[5]
        gpu_stencil = to_numpy(gpu_outputs[5])
        np.testing.assert_allclose(cpu_stencil, gpu_stencil, atol=self.ATOL)


class TestPreprocessEdgeCases:
    def test_explicit_sim_mask(self, inputs):
        """Passing a partial sim_mask reduces the number of simulation points."""
        xx, yy, grid, variogram, _, radius, stencil = inputs
        shape = xx.shape
        mask = np.zeros(shape, dtype=bool)
        mask[:shape[0]//2, :] = True  # simulate only top half

        cpu_out = _preprocess_cpu(xx, yy, grid, variogram, mask, radius, stencil)
        gpu_out = _preprocess_gpu(
            cp.asarray(xx), cp.asarray(yy), cp.asarray(grid),
            variogram, mask, radius, stencil,
        )

        cpu_inds = cpu_out[2]
        gpu_inds = to_numpy(gpu_out[2])
        assert cpu_inds.shape[0] == gpu_inds.shape[0]
        assert cpu_inds.shape[0] < shape[0] * shape[1]  # fewer than full grid

    def test_explicit_stencil(self, inputs):
        """Pre-built stencil is passed through unchanged."""
        xx, yy, grid, variogram, sim_mask, radius, _ = inputs
        custom_stencil = np.ones((5, 5), dtype=bool)

        cpu_out = _preprocess_cpu(xx, yy, grid, variogram, sim_mask, radius, custom_stencil)
        gpu_out = _preprocess_gpu(
            cp.asarray(xx), cp.asarray(yy), cp.asarray(grid),
            variogram, sim_mask, radius, custom_stencil,
        )

        np.testing.assert_array_equal(cpu_out[5], to_numpy(gpu_out[5]))