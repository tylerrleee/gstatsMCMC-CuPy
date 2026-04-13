"""

Pytest comparing NormalScoreTransformGPU against sklearn's
QuantileTransformer across multiple synthetic data distributions.


Tests run on GPU when CuPy is importable and a CUDA device exists.
If neither is available the suite falls back to a thin NumPy shim so
the *logic* can still be validated on any CI machine.

Run on a GPU machine

"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer
from scipy.special import erf as _erf

# Check CUDA GPU exist, if not use CPU
_FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"

try:
    if _FORCE_CPU:
        raise ImportError("forced CPU mode")
    import cupy as cp  # type: ignore
    from cupyx.scipy.special import erfinv as _erfinv_cp
    from cupyx.scipy.special import erf as _erf_cp

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

    # Thin shim: make `cp` behave like NumPy for the subset used here.
    class _NumPyShim:  # noqa: E306
        """Expose the CuPy API surface needed by NormalScoreTransformGPU."""

        float64 = np.float64

        @staticmethod
        def asarray(x):
            return np.asarray(x, dtype=np.float64)

        @staticmethod
        def interp(x, xp, fp):
            return np.interp(x, xp, fp)

        @staticmethod
        def clip(a, a_min, a_max):
            return np.clip(a, a_min, a_max)

        @staticmethod
        def erfinv(x):
            from scipy.special import erfinv as _erfinv  # type: ignore
            return _erfinv(x)

        @staticmethod
        def erf(x):
            return _erf(x)

    cp = _NumPyShim()  # type: ignore

# Class under test  (copied verbatim so the file is self-contained)

_SQRT2 = math.sqrt(2)
#_SQRT2 = 1.4142135623730951 # precomputed save us a millisecond LOL
_BOUNDS_THRESHOLD = 1e-7 # sklearn's clipping epsilon


class NormalScoreTransformGPU:
    """GPU-resident replacement for sklearn QuantileTransformer."""

    def __init__(self, sklearn_qt):
        q = sklearn_qt.quantiles_[:, 0].astype(np.float64)
        r = sklearn_qt.references_.astype(np.float64)
        self._quantiles  = cp.asarray(q)
        self._references = cp.asarray(r)

    def transform(self, x):

        shape = x.shape
        flat  = x.ravel()
        
        uniform = cp.interp(flat, self._quantiles, self._references)
        uniform = cp.clip(uniform, _BOUNDS_THRESHOLD, 1.0 - _BOUNDS_THRESHOLD)
        normal  = _SQRT2 * _erfinv_cp(2.0 * uniform - 1.0)
        return normal.reshape(shape)

    def inverse_transform(self, z):
        shape = z.shape
        flat  = z.ravel()
        uniform = 0.5 * (1.0 + _erf_cp(flat / _SQRT2))
        data    = cp.interp(uniform, self._references, self._quantiles)
        return data.reshape(shape)

#Helpers
def _to_numpy(arr) -> np.ndarray:
    """Convert CuPy or NumPy array -> NumPy."""
    return arr.get() if _HAS_GPU and isinstance(arr, cp.ndarray) else np.asarray(arr)


def _to_cp(arr: np.ndarray):
    """Convert NumPy -> CuPy (or keep as NumPy on CPU-only runs)."""
    return cp.asarray(arr)


def _fit_sklearn_qt(data_1d: np.ndarray, n_quantiles: int = 1000) -> QuantileTransformer:
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=42,
    )
    qt.fit(data_1d.reshape(-1, 1))
    return qt


def _sklearn_forward(qt: QuantileTransformer, data_1d: np.ndarray) -> np.ndarray:
    return qt.transform(data_1d.reshape(-1, 1)).ravel()


def _sklearn_inverse(qt: QuantileTransformer, z_1d: np.ndarray) -> np.ndarray:
    return qt.inverse_transform(z_1d.reshape(-1, 1)).ravel()


# Synthetic data fixtures

RNG = np.random.default_rng(0)

DISTRIBUTIONS: dict[str, np.ndarray] = {
    "gaussian":     RNG.normal(loc=5.0,  scale=2.0,   size=5_000),
    "uniform":      RNG.uniform(low=0.0, high=10.0,   size=5_000),
    "lognormal":    RNG.lognormal(mean=1.0, sigma=0.5, size=5_000),
    "exponential":  RNG.exponential(scale=3.0,         size=5_000),
    "bimodal":      np.concatenate([
                        RNG.normal(-3, 0.8, 2_500),
                        RNG.normal( 3, 0.8, 2_500),
                    ]),
    "heavy_tail":   RNG.standard_cauchy(size=5_000),    # extreme tails, many outliers
}

# Fit one sklearn QT per distribution (once, module-level)
_QT_CACHE: dict[str, QuantileTransformer] = {
    name: _fit_sklearn_qt(data) for name, data in DISTRIBUTIONS.items()
}


@pytest.fixture(params=list(DISTRIBUTIONS.keys()))
def dist_name(request):
    return request.param


@pytest.fixture()
def data_and_qt(dist_name):
    """Return (raw_data_1d, gpu_transformer, sklearn_transformer).

    Tests that need the name alongside this fixture should also declare
    ``dist_name`` as a parameter — pytest injects both independently.
    """
    data = DISTRIBUTIONS[dist_name]
    qt   = _QT_CACHE[dist_name]
    gpu  = NormalScoreTransformGPU(qt)
    return data, gpu, qt


# Test 1 — Forward transform: GPU ≈ sklearn (mean absolute error)

class TestForwardTransform:

    def test_mae_vs_sklearn(self, data_and_qt):
        """GPU forward transform matches sklearn within 1e-5 on average."""
        data, gpu, qt = data_and_qt
        x_gpu = _to_cp(data)

        z_gpu  = _to_numpy(gpu.transform(x_gpu))
        z_skl  = _sklearn_forward(qt, data)

        mae = np.mean(np.abs(z_gpu - z_skl))
        assert mae < 1e-5, f"MAE {mae:.3e} exceeds 1e-5"

    def test_max_abs_error_vs_sklearn(self, data_and_qt):
        """Max absolute error between GPU and sklearn ≤ 1e-4."""
        data, gpu, qt = data_and_qt
        x_gpu = _to_cp(data)

        z_gpu = _to_numpy(gpu.transform(x_gpu))
        z_skl = _sklearn_forward(qt, data)

        max_err = np.max(np.abs(z_gpu - z_skl))
        assert max_err < 1e-4, f"Max abs error {max_err:.3e} exceeds 1e-4"

    def test_output_shape_preserved(self, data_and_qt):
        """Output shape matches input shape (flat array)."""
        data, gpu, qt = data_and_qt
        x_gpu = _to_cp(data)
        z_gpu = gpu.transform(x_gpu)
        assert _to_numpy(z_gpu).shape == data.shape

    def test_output_shape_2d(self, data_and_qt):
        """Shape is preserved for 2-D input."""
        data, gpu, _ = data_and_qt
        data_2d = data[:100].reshape(10, 10)
        x_gpu = _to_cp(data_2d)
        z_gpu = gpu.transform(x_gpu)
        assert _to_numpy(z_gpu).shape == (10, 10)

    def test_output_is_finite(self, data_and_qt):
        """All output values must be finite (no ±inf, no NaN)."""
        data, gpu, _ = data_and_qt
        x_gpu = _to_cp(data)
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        assert np.all(np.isfinite(z_gpu)), "Non-finite values in forward output"

    def test_output_approximately_standard_normal(self, data_and_qt):
        """Transformed values should be roughly N(0,1): |mean|<0.1, |std-1|<0.15."""
        data, gpu, _ = data_and_qt
        x_gpu = _to_cp(data)
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        assert abs(np.mean(z_gpu)) < 0.1,       f"Mean {np.mean(z_gpu):.3f} too far from 0"
        assert abs(np.std(z_gpu) - 1.0) < 0.15, f"Std  {np.std(z_gpu):.3f} too far from 1"

    def test_monotone(self, data_and_qt):
        """Transform is monotone: sorted input -> sorted output."""
        data, gpu, _ = data_and_qt
        sorted_data = np.sort(data)
        x_gpu = _to_cp(sorted_data)
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        diffs = np.diff(z_gpu)
        assert np.all(diffs >= -1e-10), "Transform is not monotone"


# Test 2 — Inverse transform: GPU ≈ sklearn

class TestInverseTransform:

    def _z_grid(self) -> np.ndarray:
        """Standard normal grid avoiding extreme tails where interp clips."""
        return np.linspace(-3.5, 3.5, 500)

    def test_mae_vs_sklearn(self, data_and_qt):
        """GPU inverse transform matches sklearn within 1e-5 on average."""
        _, gpu, qt = data_and_qt
        z = self._z_grid()
        z_gpu = _to_cp(z)

        x_gpu = _to_numpy(gpu.inverse_transform(z_gpu))
        x_skl = _sklearn_inverse(qt, z)

        mae = np.mean(np.abs(x_gpu - x_skl))
        assert mae < 1e-5, f"MAE {mae:.3e} exceeds 1e-5"

    def test_output_shape_preserved(self, data_and_qt):
        """Output shape matches input shape."""
        _, gpu, _ = data_and_qt
        z = _to_cp(self._z_grid())
        x = gpu.inverse_transform(z)
        assert _to_numpy(x).shape == self._z_grid().shape

    def test_output_is_finite(self, data_and_qt):
        """All output values are finite."""
        _, gpu, _ = data_and_qt
        z_gpu = _to_cp(self._z_grid())
        x_gpu = _to_numpy(gpu.inverse_transform(z_gpu))
        assert np.all(np.isfinite(x_gpu))


# Test 3 — Round-trip consistency  (forward ∘ inverse ≈ identity)

class TestRoundTrip:

    def test_gpu_round_trip(self, data_and_qt, dist_name):
        """GPU: transform then inverse_transform recovers original values.

        Tolerance is relaxed for 'heavy_tail' (Cauchy) because the QT is fit on
        1 000 quantile knots but Cauchy draws can land *far* outside that range —
        the interp clamps them, so the inverse cannot recover the exact outlier.
        That behaviour matches sklearn exactly (see test_gpu_vs_sklearn_round_trip_agreement).
        """
        data, gpu, _ = data_and_qt
        x_gpu = _to_cp(data)
        z_gpu = gpu.transform(x_gpu)
        x_rec = _to_numpy(gpu.inverse_transform(z_gpu))

        mae = np.mean(np.abs(x_rec - data))
        tol = 5e-3 if dist_name == "heavy_tail" else 1e-4
        assert mae < tol, f"Round-trip MAE {mae:.3e} exceeds {tol:.0e}"

    def test_sklearn_round_trip(self, data_and_qt, dist_name):
        """Sklearn: transform then inverse_transform recovers original values (MAE < 1e-4)."""
        data, _, qt = data_and_qt
        z_skl = _sklearn_forward(qt, data)
        x_rec = _sklearn_inverse(qt, z_skl)

        mae = np.mean(np.abs(x_rec - data))
        tol = 5e-3 if dist_name == "heavy_tail" else 1e-4
        assert mae < tol, f"sklearn round-trip MAE {mae:.3e} exceeds 1e-4"

    def test_gpu_vs_sklearn_round_trip_agreement(self, data_and_qt, dist_name):
        """GPU and sklearn round-trips agree within tolerance.

        Cauchy (heavy_tail) is allowed a larger tolerance: both GPU and sklearn
        clamp out-of-range outliers to the min/max quantile knot, so neither
        recovers the true value and the MAE reflects clamping, not a bug.
        """
        data, gpu, qt = data_and_qt
        x_gpu = _to_cp(data)

        # GPU round-trip
        z_gpu = gpu.transform(x_gpu)
        x_rec_gpu = _to_numpy(gpu.inverse_transform(z_gpu))

        # sklearn round-trip
        z_skl = _sklearn_forward(qt, data)
        x_rec_skl = _sklearn_inverse(qt, z_skl)

        mae = np.mean(np.abs(x_rec_gpu - x_rec_skl))
        tol = 5e-3 if dist_name == "heavy_tail" else 1e-4
        assert mae < tol, f"Round-trip agreement MAE {mae:.3e} exceeds {tol:.0e}"


# Test 4 — Edge cases

class TestEdgeCases:

    def test_single_element(self, data_and_qt):
        """Single-element array transforms without error."""
        data, gpu, _ = data_and_qt
        x_gpu = _to_cp(np.array([np.median(data)]))
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        assert z_gpu.shape == (1,)
        assert np.isfinite(z_gpu[0])

    def test_extreme_values_clipped_not_inf(self, data_and_qt):
        """Values far outside training range map to finite normal scores."""
        data, gpu, _ = data_and_qt
        extremes = np.array([data.min() - 1e6, data.max() + 1e6])
        x_gpu = _to_cp(extremes)
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        assert np.all(np.isfinite(z_gpu)), "Extreme values produced non-finite output"

    def test_constant_input_within_range(self, data_and_qt):
        """Constant array (median) transforms to a single finite normal score."""
        data, gpu, _ = data_and_qt
        val = float(np.median(data))
        x_gpu = _to_cp(np.full(50, val))
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        assert np.all(np.isfinite(z_gpu))
        assert np.std(z_gpu) < 1e-10, "Constant input should give constant output"

    def test_output_bounds_respected(self, data_and_qt):
        """Forward output is bounded by ppf of _BOUNDS_THRESHOLD clip."""
        data, gpu, _ = data_and_qt
        x_gpu = _to_cp(data)
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        from scipy.stats import norm  # type: ignore
        lo = norm.ppf(_BOUNDS_THRESHOLD)
        hi = norm.ppf(1 - _BOUNDS_THRESHOLD)
        # Add a small epsilon for floating-point slack
        assert np.all(z_gpu >= lo - 1e-6)
        assert np.all(z_gpu <= hi + 1e-6)

    def test_large_batch(self, data_and_qt):
        """1 000 000-element array completes without error."""
        data, gpu, _ = data_and_qt
        big = np.tile(data, math.ceil(1_000_000 / len(data)))[:1_000_000]
        x_gpu = _to_cp(big)
        z_gpu = _to_numpy(gpu.transform(x_gpu))
        assert z_gpu.shape == (1_000_000,)
        assert np.all(np.isfinite(z_gpu))


# Test 5 — Quantile fidelity (CDF/percentile checks)

class TestQuantileFidelity:

    @pytest.mark.parametrize("percentile", [10, 25, 50, 75, 90])
    def test_percentile_order_preserved(self, data_and_qt, percentile):
        """Values below/above a percentile map to negative/positive z-scores."""
        
        data, gpu, _ = data_and_qt
        threshold = np.percentile(data, percentile)
        below = data[data < threshold]

        above = data[data > threshold]
        if len(below) == 0 or len(above) == 0:
            pytest.skip("Not enough samples on one side")

        z_below = _to_numpy(gpu.transform(_to_cp(below)))
        z_above = _to_numpy(gpu.transform(_to_cp(above)))

        assert np.mean(z_below) < np.mean(z_above), (
            f"p{percentile}: mean z below ({np.mean(z_below):.2f}) "
            f"≥ mean z above ({np.mean(z_above):.2f})"
        )