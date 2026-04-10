"""
test_mcmc_comparison.py
=======================
Benchmarking harness that runs the same MCMC workflow three ways:

  A) CPU-only:   MCMC.chain_sgs          (original NumPy/sklearn)
  B) GPU v1:     MCMC_cu.chain_sgs_gpu   (old sgs_gpu — full preprocess every iter)
  C) GPU v2:     MCMC_cu.chain_sgs_gpu   (with SGS_MCMC context — no redundant setup)

Replicates the T4_GPU_SmallScaleChain notebook workflow exactly.
Run on a machine with the project code, data files, and a CUDA GPU.

Usage:
    python test_mcmc_comparison.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import time
import json
import os
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.preprocessing import QuantileTransformer
from GPU import MCMC_cu, QuantileTransformer_gpu

from gstatsMCMC import MCMC
    
DATA_CSV        = '../data/Supprt_Force (1).csv'
INITIAL_BED_NPY = '../data/bed_1000k.npy'
RNG_SEED        = 81978947
RESOLUTION      = 500

# Variogram params — use your saved values, or let the script compute them
# Set to None to compute from data; set to a list to skip variogram fitting
V1_P_OVERRIDE   = None  

# Test iterations — keep small for benchmarking, increase for validation
N_ITER_BENCHMARK = 500    # quick timing comparison
N_ITER_VALIDATE  = 2000   # longer run to compare loss trajectories

# Which tests to run
RUN_CPU      = True   # (A) original CPU chain — can be very slow
RUN_GPU_V1   = True   # (B) GPU with old sgs_gpu (full preprocess each iter)
RUN_GPU_V2   = True   # (C) GPU with SGS_MCMC context

# Block sizes (matching notebook)
MIN_BLOCK_X, MAX_BLOCK_X = 5, 20
MIN_BLOCK_Y, MAX_BLOCK_Y = 5, 20

# SGS params (matching notebook)
SGS_NEIGHBORS = 48
SGS_RADIUS    = 30e3
SIGMA_MC      = 5

def load_data():
    """Load and prepare all data arrays. Returns a dict."""
    
    print("Loading data...")
    df = pd.read_csv(DATA_CSV)

    x_uniq = np.unique(df.x)
    y_uniq = np.unique(df.y)
    xx, yy = np.meshgrid(x_uniq, y_uniq)
    rows, cols = xx.shape

    dhdt                 = df['dhdt'].values.reshape(xx.shape)
    smb                  = df['smb'].values.reshape(xx.shape)
    velx                 = df['velx'].values.reshape(xx.shape)
    vely                 = df['vely'].values.reshape(xx.shape)
    bedmap_mask          = df['bedmap_mask'].values.reshape(xx.shape)
    bedmachine_thickness = df['bedmachine_thickness'].values.reshape(xx.shape)
    bedmap_surf          = df['bedmap_surf'].values.reshape(xx.shape)
    highvel_mask         = df['highvel_mask'].values.reshape(xx.shape)
    bedmap_bed           = df['bedmap_bed'].values.reshape(xx.shape)

    # Conditioning data
    cond_bed = np.where(bedmap_mask == 1,
                        df['bed'].values.reshape(xx.shape),
                        bedmap_bed)
    df['cond_bed'] = cond_bed.flatten()
    data_mask = ~np.isnan(cond_bed)

    # Initial bed + trend
    initial_bed = np.load(INITIAL_BED_NPY)
    trend = sp.ndimage.gaussian_filter(initial_bed, sigma=10)

    grounded_ice_mask = (bedmap_mask == 1)

    return {
        'df': df, 'xx': xx, 'yy': yy,
        'dhdt': dhdt, 'smb': smb, 'velx': velx, 'vely': vely,
        'bedmap_mask': bedmap_mask, 'bedmap_surf': bedmap_surf,
        'highvel_mask': highvel_mask, 'bedmap_bed': bedmap_bed,
        'cond_bed': cond_bed, 'data_mask': data_mask,
        'initial_bed': initial_bed, 'trend': trend,
        'grounded_ice_mask': grounded_ice_mask,
        'rows': xx.shape[0], 'cols': xx.shape[1],
    }

def fit_variogram_and_transform(data_dict):
    """
    Fit variogram and normal score transform.
    Returns V1_p (variogram params) and sklearn_qt (fitted QuantileTransformer).
    """
    df    = data_dict['df']
    trend = data_dict['trend']
    initial_bed = data_dict['initial_bed']

    # Detrended residuals
    df['cond_bed_residual'] = df['cond_bed'].values - trend.flatten()
    data_for_distribution = (initial_bed - trend).reshape((-1, 1))

    sklearn_qt = QuantileTransformer(
        n_quantiles=1000,
        output_distribution="normal",
        subsample=None,
        random_state=RNG_SEED
    ).fit(data_for_distribution)

    # Transform conditioning data for variogram fitting
    data = df['cond_bed_residual'].values.reshape(-1, 1)
    transformed_data = sklearn_qt.transform(data)
    df['Nbed_residual'] = transformed_data

    # Variogram
    if V1_P_OVERRIDE is not None:
        V1_p = V1_P_OVERRIDE
        print(f"  Using override variogram params: {V1_p}")
    else:
        import skgstat as skg
        print("  Fitting variogram (this may take a minute)...")
        df_sampled = df[df["cond_bed_residual"].isnull() == False]
        df_sampled = df_sampled[df_sampled["bedmap_mask"] == 1]

        coords = df_sampled[['x', 'y']].values
        values = df_sampled['Nbed_residual']

        V1 = skg.Variogram(coords, values, bin_func='even',
                           n_lags=70, maxlag=50000,
                           normalize=False, model='matern')
        V1_p = V1.parameters
        print(f"  Fitted variogram params: {V1_p}")

    return V1_p, sklearn_qt


def configure_chain(chain_obj, data_dict, V1_p, nst_trans_obj):
    """Apply all the set_*() calls matching the notebook."""
    chain_obj.set_update_region(True, data_dict['highvel_mask'])
    chain_obj.set_loss_type(sigma_mc=SIGMA_MC, massConvInRegion=True)
    chain_obj.set_block_sizes(MIN_BLOCK_X, MAX_BLOCK_X, MIN_BLOCK_Y, MAX_BLOCK_Y)
    chain_obj.set_normal_transformation(nst_trans_obj, do_transform=True)
    chain_obj.set_trend(trend=data_dict['trend'], detrend_map=True)
    chain_obj.set_variogram('Matern', V1_p[0], V1_p[1], 0,
                            isotropic=True, vario_smoothness=V1_p[2])
    chain_obj.set_sgs_param(SGS_NEIGHBORS, SGS_RADIUS, sgs_rand_dropout_on=False)
    chain_obj.set_random_generator(rng_seed=RNG_SEED)


def run_chain(chain_obj, n_iter, label):
    """Run a chain, return timing + loss array."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}  ({n_iter} iterations)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = chain_obj.run(
        n_iter=n_iter,
        info_per_iter=max(1, n_iter // 5),
        only_save_last_bed=True,
        plot=False,
        progress_bar=True
    )
    elapsed = time.perf_counter() - t0

    # result = (last_bed, loss_mc, loss_data, loss, steps, resampled, blocks)
    last_bed       = result[0]
    loss_mc_cache  = result[1]
    loss_cache     = result[3]
    step_cache     = result[4]

    # Pull to numpy for comparison
    if hasattr(loss_cache, 'get'):
        loss_np    = loss_cache.get()
        step_np    = step_cache.get()
        loss_mc_np = loss_mc_cache.get()
    else:
        loss_np    = np.asarray(loss_cache)
        step_np    = np.asarray(step_cache)
        loss_mc_np = np.asarray(loss_mc_cache)

    acc_rate = np.sum(step_np) / max(1, n_iter - 1)

    print(f"\n  {label} results:")
    print(f"    Wall time:       {elapsed:.2f} s")
    print(f"    Iterations/sec:  {(n_iter - 1) / elapsed:.2f}")
    print(f"    Final loss:      {loss_np[-1]:.6e}")
    print(f"    Final MC loss:   {loss_mc_np[-1]:.6e}")
    print(f"    Acceptance rate: {acc_rate:.4f}")

    return {
        'label':     label,
        'elapsed':   elapsed,
        'iter_per_s': (n_iter - 1) / elapsed,
        'loss':      loss_np,
        'loss_mc':   loss_mc_np,
        'steps':     step_np,
        'acc_rate':  acc_rate,
        'final_loss': float(loss_np[-1]),
    }


def compare_results(results):
    """Print a comparison table and optionally save a plot."""

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Label':<25} {'Time (s)':>10} {'it/s':>10} {'Final Loss':>14} {'Acc Rate':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")

    for r in results:
        print(f"  {r['label']:<25} {r['elapsed']:>10.2f} {r['iter_per_s']:>10.2f} "
              f"{r['final_loss']:>14.6e} {r['acc_rate']:>10.4f}")

    # Speedup relative to first result
    if len(results) > 1:
        base = results[0]['elapsed']
        print(f"\n  Speedups vs {results[0]['label']}:")
        for r in results[1:]:
            speedup = base / r['elapsed']
            print(f"    {r['label']}: {speedup:.2f}x")

    # Save loss curves
    try:

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss trajectories
        for r in results:
            axes[0].plot(r['loss'], label=r['label'], alpha=0.8)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Trajectory Comparison')
        axes[0].legend()
        axes[0].set_yscale('log')

        # Bar chart of it/s
        labels = [r['label'] for r in results]
        speeds = [r['iter_per_s'] for r in results]
        bars = axes[1].bar(labels, speeds, color=['#2196F3', '#FF9800', '#4CAF50'][:len(results)])
        axes[1].set_ylabel('Iterations / second')
        axes[1].set_title('Throughput Comparison')
        for bar, speed in zip(bars, speeds):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{speed:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('mcmc_benchmark_comparison.png', dpi=150)
        print(f"\n  Plot saved to: mcmc_benchmark_comparison.png")
    except Exception as e:
        print(f"\n  (Skipping plot: {e})")

    # Save JSON summary
    summary = []
    for r in results:
        summary.append({
            'label':      r['label'],
            'elapsed_s':  r['elapsed'],
            'iter_per_s': r['iter_per_s'],
            'final_loss': r['final_loss'],
            'acc_rate':   r['acc_rate'],
        })
    with open('mcmc_benchmark_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to: mcmc_benchmark_results.json")


def main():
    data = load_data()
    V1_p, sklearn_qt = fit_variogram_and_transform(data)

    print(f"\n  Grid shape: {data['xx'].shape}")
    print(f"  Variogram:  {V1_p}")

    results = []


    if RUN_CPU:

        cpu_chain = MCMC.chain_sgs(
            data['xx'], data['yy'], data['initial_bed'],
            data['bedmap_surf'], data['velx'], data['vely'],
            data['dhdt'], data['smb'], data['cond_bed'],
            data['data_mask'], data['grounded_ice_mask'], RESOLUTION
        )
        # CPU chain uses sklearn QuantileTransformer directly
        configure_chain(cpu_chain, data, V1_p, sklearn_qt)
        results.append(run_chain(cpu_chain, N_ITER_BENCHMARK, "CPU (MCMC.chain_sgs)"))


    if RUN_GPU_V1:
        try:
            nst_gpu = QuantileTransformer_gpu.NormalScoreTransformGPU(sklearn_qt)

            gpu_v1_chain = MCMC_cu.chain_sgs_gpu(
                data['xx'], data['yy'], data['initial_bed'],
                data['bedmap_surf'], data['velx'], data['vely'],
                data['dhdt'], data['smb'], data['cond_bed'],
                data['data_mask'], data['grounded_ice_mask'], RESOLUTION
            )
            configure_chain(gpu_v1_chain, data, V1_p, nst_gpu)
            results.append(run_chain(gpu_v1_chain, N_ITER_BENCHMARK,
                                     "GPU"))
        except ImportError:
            print("\n  GPU — MCMC_cu not found.")

if __name__ == '__main__':
    main()
