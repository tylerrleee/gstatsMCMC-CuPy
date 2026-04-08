import cupy as cp
from cupyx.scipy.special import erfinv, erf
_SQRT2 = 1.4142135623730951        # math.sqrt(2), precomputed
_BOUNDS_THRESHOLD = 1e-7           # sklearn's clipping epsilon
import numpy as np

class NormalScoreTransformGPU:
    """GPU-resident replacement for sklearn QuantileTransformer.

    Extracts the two lookup arrays from a *fitted* sklearn
    QuantileTransformer and stores them as CuPy arrays.  Forward
    and inverse transforms then run entirely on the GPU using
    cp.interp + elementary math (erf / erfinv) — no CPU round-trip,
    no sklearn call, no cudaMemcpy per iteration.

    Usage
    -----
    # one-time setup (in set_normal_transformation or pre-loop):
    gpu_nst = NormalScoreTransformGPU(sklearn_qt)

    # per-iteration (drop-in for nst_trans.transform / inverse_transform):
    z = gpu_nst.transform(bed_c)          # cp.ndarray -> cp.ndarray
    bed = gpu_nst.inverse_transform(z)    # cp.ndarray -> cp.ndarray
    """

    def __init__(self, sklearn_qt):
        """Extract lookup tables from a fitted QuantileTransformer.

        Args:
            sklearn_qt: A fitted sklearn.preprocessing.QuantileTransformer
                with output_distribution='normal' and a single feature column.
        """
        # sklearn stores:
        #   quantiles_  : shape (n_quantiles, n_features) — data values at each quantile level
        #   references_ : shape (n_quantiles,)            — uniform quantile levels in [0, 1]
        #
        # Forward transform:  data -> uniform (interp on quantiles_ -> references_)
        #                     uniform -> normal (ppf, clipped to avoid ±inf)
        #
        # Inverse transform:  normal -> uniform (cdf)
        #                     uniform -> data (interp on references_ -> quantiles_)

        q = sklearn_qt.quantiles_[:, 0].astype(np.float64)   # (n_quantiles,)
        r = sklearn_qt.references_.astype(np.float64)         # (n_quantiles,)

        # Upload once — these never change
        self._quantiles  = cp.asarray(q)    # xp for forward interp
        self._references = cp.asarray(r)    # fp for forward interp

    def transform(self, x):
        """Forward: data values -> normal scores.  Fully on GPU.

        Args:
            x (cp.ndarray): Any-shape CuPy array of data values.

        Returns:
            cp.ndarray: Normal-score-transformed values, same shape as x.
        """
        x = cp.asarray(x)
            
        
        shape = x.shape
        flat = x.ravel()

        # Step 1: data -> uniform quantile level via piecewise-linear interp
        # cp.interp(x, xp, fp) — xp must be increasing, which quantiles_ is.
        # NaN inputs pass through as NaN (same as np.interp).
        uniform = cp.interp(flat, self._quantiles, self._references)

        # Step 2: clip to [eps, 1-eps] to avoid ±inf from ppf
        uniform = cp.clip(uniform, _BOUNDS_THRESHOLD, 1.0 - _BOUNDS_THRESHOLD)

        # Step 3: uniform -> normal via ppf(u) = sqrt(2) * erfinv(2u - 1)
        normal = _SQRT2 * erfinv(2.0 * uniform - 1.0)   # CuPy exposes erfinv natively (cupyx.scipy.special)

        return normal.reshape(shape)

    def inverse_transform(self, z):
        """Inverse: normal scores -> data values.  Fully on GPU.

        Args:
            z (cp.ndarray): Any-shape CuPy array of normal-score values.

        Returns:
            cp.ndarray: Original-scale data values, same shape as z.
        """
        z = cp.asarray(z)
        shape = z.shape
        flat = z.ravel()

        # Step 1: normal -> uniform via cdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
        uniform = 0.5 * (1.0 + erf(flat / _SQRT2))

        # Step 2: uniform -> data via interp on references_ -> quantiles_
        # references_ is increasing [0..1], so cp.interp works directly.
        # Out-of-range uniforms (≈0 or ≈1) clamp to the min/max quantile,
        # exactly matching sklearn's behavior.
        data = cp.interp(uniform, self._references, self._quantiles)

        return data.reshape(shape)