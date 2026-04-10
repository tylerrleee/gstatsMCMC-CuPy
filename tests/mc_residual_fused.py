"""
Single CUDA kernel for mass conservation residual.

Replaces 6+ kernel launches with 1:
    thick = surf - bed
    flux_x = velx * thick   →  d(flux_x)/dx  via np.gradient axis=1
    flux_y = vely * thick   →  d(flux_y)/dy  via np.gradient axis=0
    res = d(flux_x)/dx + d(flux_y)/dy + dhdt - smb

Boundary handling matches np.gradient / cp.gradient exactly:
    left/top edge:    forward  difference   (f[1] - f[0]) / h
    interior:         central  difference   (f[i+1] - f[i-1]) / (2h)
    right/bottom edge: backward difference  (f[-1] - f[-2]) / h

Usage:
    from mc_residual_fused import get_mass_conservation_residual_fused
    res = get_mass_conservation_residual_fused(bed, surf, velx, vely, dhdt, smb, resolution)
"""

import cupy as cp
import matplotlib.pyplot as plt
import time
import numpy as np

# ── Raw CUDA kernel ──────────────────────────────────────────
_mc_residual_kernel = cp.RawKernel(r'''
extern "C" __global__
void mc_residual(
    const double* __restrict__ bed,
    const double* __restrict__ surf,
    const double* __restrict__ velx,
    const double* __restrict__ vely,
    const double* __restrict__ dhdt,
    const double* __restrict__ smb,
    double* __restrict__ res,
    const int rows,
    const int cols,
    const double inv_h,      // 1.0 / resolution
    const double inv_2h      // 1.0 / (2.0 * resolution)
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int r = idx / cols;
    int c = idx % cols;

    // ── thickness at this cell ──────────────────────────────
    double thick = surf[idx] - bed[idx];

    // ── flux_x = velx * thick  →  d(flux_x)/dx along axis=1 (columns) ──
    double dx_val;
    if (cols == 1) {
        dx_val = 0.0;
    } else if (c == 0) {
        // forward difference
        int idx_r = r * cols + 1;
        double fx_here = velx[idx] * thick;
        double fx_right = velx[idx_r] * (surf[idx_r] - bed[idx_r]);
        dx_val = (fx_right - fx_here) * inv_h;
    } else if (c == cols - 1) {
        // backward difference
        int idx_l = r * cols + (c - 1);
        double fx_here = velx[idx] * thick;
        double fx_left = velx[idx_l] * (surf[idx_l] - bed[idx_l]);
        dx_val = (fx_here - fx_left) * inv_h;
    } else {
        // central difference
        int idx_l = r * cols + (c - 1);
        int idx_r = r * cols + (c + 1);
        double fx_left  = velx[idx_l] * (surf[idx_l] - bed[idx_l]);
        double fx_right = velx[idx_r] * (surf[idx_r] - bed[idx_r]);
        dx_val = (fx_right - fx_left) * inv_2h;
    }

    // ── flux_y = vely * thick  →  d(flux_y)/dy along axis=0 (rows) ──
    double dy_val;
    if (rows == 1) {
        dy_val = 0.0;
    } else if (r == 0) {
        // forward difference
        int idx_d = (1) * cols + c;
        double fy_here = vely[idx] * thick;
        double fy_down = vely[idx_d] * (surf[idx_d] - bed[idx_d]);
        dy_val = (fy_down - fy_here) * inv_h;
    } else if (r == rows - 1) {
        // backward difference
        int idx_u = (r - 1) * cols + c;
        double fy_here = vely[idx] * thick;
        double fy_up   = vely[idx_u] * (surf[idx_u] - bed[idx_u]);
        dy_val = (fy_here - fy_up) * inv_h;
    } else {
        // central difference
        int idx_u = (r - 1) * cols + c;
        int idx_d = (r + 1) * cols + c;
        double fy_up   = vely[idx_u] * (surf[idx_u] - bed[idx_u]);
        double fy_down = vely[idx_d] * (surf[idx_d] - bed[idx_d]);
        dy_val = (fy_down - fy_up) * inv_2h;
    }

    // ── final residual ──────────────────────────────────────
    res[idx] = dx_val + dy_val + dhdt[idx] - smb[idx];
}
''', 'mc_residual')


def get_mass_conservation_residual_fused(bed, surf, velx, vely, dhdt, smb, resolution):
    """
    Drop-in replacement for get_mass_conservation_residual_GPU.
    Single kernel launch instead of 6+.

    All inputs must be 2D CuPy float64 arrays of the same shape.
    resolution is a Python float or int.
    Returns a 2D CuPy float64 array of the same shape.
    """
    rows, cols = bed.shape
    n = rows * cols

    res = cp.empty_like(bed)

    inv_h  = 1.0 / float(resolution)
    inv_2h = 1.0 / (2.0 * float(resolution))

    threads = 256
    blocks = (n + threads - 1) // threads

    import numpy as np
    _mc_residual_kernel(
        (blocks,), (threads,),
        (bed, surf, velx, vely, dhdt, smb, res,
         np.int32(rows), np.int32(cols),
         np.float64(inv_h), np.float64(inv_2h))
    )

    return res


def get_mass_conservation_residual_fused_local(
    bed_local, surf_local, velx_local, vely_local,
    dhdt_local, smb_local, resolution
):
    """
    Same as above but for small local blocks extracted from the full grid.
    Ensures contiguous memory layout before launching the kernel.
    """
    # Slices from CuPy arrays may not be contiguous
    bed_local  = cp.ascontiguousarray(bed_local)
    surf_local = cp.ascontiguousarray(surf_local)
    velx_local = cp.ascontiguousarray(velx_local)
    vely_local = cp.ascontiguousarray(vely_local)
    dhdt_local = cp.ascontiguousarray(dhdt_local)
    smb_local  = cp.ascontiguousarray(smb_local)

    return get_mass_conservation_residual_fused(
        bed_local, surf_local, velx_local, vely_local,
        dhdt_local, smb_local, resolution
    )


# ── Verification & Plotting ──────────────────────────────────
if __name__ == '__main__':
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  Fused MC Residual Kernel — Verification & Benchmark")
    print("=" * 60)

    # ── Reference implementation (original 6-kernel version) ─
    def get_mass_conservation_residual_GPU_reference(bed, surf, velx, vely, dhdt, smb, resolution):
        thick = surf - bed
        dx = cp.gradient(velx * thick, resolution, axis=1)
        dy = cp.gradient(vely * thick, resolution, axis=0)
        return dx + dy + dhdt - smb

    # Lists to store benchmarking data for plotting
    grid_labels = []
    ref_times_us = []
    fused_times_us = []

    # Test multiple grid sizes
    for shape_label, shape in [("SmallScale block\n(15x15)", (15, 15)),
                                ("LargeScale grid\n(50x50)", (50, 50)),
                                ("Full grid\n(2000x2000)", (2000, 2000))]:
        print(f"\n--- {shape_label.replace(chr(10), ' ')} ---")
        rows, cols = shape
        rng = np.random.default_rng(42)

        bed  = cp.asarray(rng.uniform(-2000, 500, (rows, cols)))
        surf = bed + cp.asarray(rng.uniform(100, 3000, (rows, cols)))
        velx = cp.asarray(rng.uniform(-500, 500, (rows, cols)))
        vely = cp.asarray(rng.uniform(-500, 500, (rows, cols)))
        dhdt = cp.asarray(rng.uniform(-10, 10, (rows, cols)))
        smb  = cp.asarray(rng.uniform(-5, 5, (rows, cols)))
        resolution = 500.0

        # Correctness check
        ref = get_mass_conservation_residual_GPU_reference(bed, surf, velx, vely, dhdt, smb, resolution)
        fused = get_mass_conservation_residual_fused(bed, surf, velx, vely, dhdt, smb, resolution)

        max_err = float(cp.max(cp.abs(ref - fused)))
        rel_err = float(cp.max(cp.abs(ref - fused) / (cp.abs(ref) + 1e-30)))
        print(f"  Max absolute error: {max_err:.2e}")
        print(f"  Max relative error: {rel_err:.2e}")
        assert max_err < 1e-10, f"FAILED: max error {max_err} too large"
        print(f"  ✓ Correctness verified")

        # Benchmark
        cp.cuda.Device().synchronize()
        n_warmup = 50
        n_bench = 500

        for _ in range(n_warmup):
            get_mass_conservation_residual_GPU_reference(bed, surf, velx, vely, dhdt, smb, resolution)
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(n_bench):
            get_mass_conservation_residual_GPU_reference(bed, surf, velx, vely, dhdt, smb, resolution)
        cp.cuda.Device().synchronize()
        t_ref = (time.perf_counter() - t0) / n_bench

        for _ in range(n_warmup):
            get_mass_conservation_residual_fused(bed, surf, velx, vely, dhdt, smb, resolution)
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(n_bench):
            get_mass_conservation_residual_fused(bed, surf, velx, vely, dhdt, smb, resolution)
        cp.cuda.Device().synchronize()
        t_fused = (time.perf_counter() - t0) / n_bench

        speedup = t_ref / t_fused
        t_ref_us = t_ref * 1e6
        t_fused_us = t_fused * 1e6
        
        print(f"  Reference (6 kernels): {t_ref_us:8.1f} µs")
        print(f"  Fused (1 kernel):      {t_fused_us:8.1f} µs")
        print(f"  Speedup: {speedup:.2f}x")

        # Store data for plotting
        grid_labels.append(shape_label)
        ref_times_us.append(t_ref_us)
        fused_times_us.append(t_fused_us)

    # Local block version verification
    print(f"\n--- Local block version (slice from 500x500 grid) ---")
    rows, cols = 500, 500
    rng = np.random.default_rng(99)
    bed  = cp.asarray(rng.uniform(-2000, 500, (rows, cols)))
    surf = bed + cp.asarray(rng.uniform(100, 3000, (rows, cols)))
    velx = cp.asarray(rng.uniform(-500, 500, (rows, cols)))
    vely = cp.asarray(rng.uniform(-500, 500, (rows, cols)))
    dhdt = cp.asarray(rng.uniform(-10, 10, (rows, cols)))
    smb  = cp.asarray(rng.uniform(-5, 5, (rows, cols)))

    bxmin, bxmax, bymin, bymax = 50, 65, 80, 100
    pad_x1, pad_x2 = max(0, bxmin-1), min(rows, bxmax+1)
    pad_y1, pad_y2 = max(0, bymin-1), min(cols, bymax+1)

    ref_local = get_mass_conservation_residual_GPU_reference(
        bed[pad_x1:pad_x2, pad_y1:pad_y2], surf[pad_x1:pad_x2, pad_y1:pad_y2],
        velx[pad_x1:pad_x2, pad_y1:pad_y2], vely[pad_x1:pad_x2, pad_y1:pad_y2],
        dhdt[pad_x1:pad_x2, pad_y1:pad_y2], smb[pad_x1:pad_x2, pad_y1:pad_y2], 500.0
    )
    fused_local = get_mass_conservation_residual_fused_local(
        bed[pad_x1:pad_x2, pad_y1:pad_y2], surf[pad_x1:pad_x2, pad_y1:pad_y2],
        velx[pad_x1:pad_x2, pad_y1:pad_y2], vely[pad_x1:pad_x2, pad_y1:pad_y2],
        dhdt[pad_x1:pad_x2, pad_y1:pad_y2], smb[pad_x1:pad_x2, pad_y1:pad_y2], 500.0
    )
    max_err = float(cp.max(cp.abs(ref_local - fused_local)))
    assert max_err < 1e-10, f"FAILED: max error {max_err} too large"
    print(f"  ✓ Local block version verified")

    print(f"\n{'='*60}")
    print(f"  All tests passed. Generating plot...")
    print(f"{'='*60}")

    # ── Matplotlib Chart Generation ──────────────────────────────
    x = np.arange(len(grid_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bars
    rects1 = ax.bar(x - width/2, ref_times_us, width, label='Reference (6 kernels)', color='#E24A33')
    rects2 = ax.bar(x + width/2, fused_times_us, width, label='Fused (1 kernel)', color='#348ABD')

    # Formatting and labels
    ax.set_ylabel('Execution Time (µs)')
    ax.set_title('GPU Performance: Reference vs. Fused Kernel')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_labels)
    ax.legend()
    
    # Using a log scale if the difference between small and full grid times is massive
    ax.set_yscale('log')

    # Add exact values on top of the bars
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    # Calculate and annotate speedup percentages
    for i in range(len(grid_labels)):
        speedup_val = ref_times_us[i] / fused_times_us[i]
        ax.text(x[i], max(ref_times_us[i], fused_times_us[i]) * 1.5, 
                f'{speedup_val:.2f}x Speedup', ha='center', va='bottom', 
                fontweight='bold', color='black')

    # Ensure layout fits well and display
    fig.tight_layout()
    
    # Save the plot to disk before showing it
    plt.savefig(f'kernel_performance_{time.time()}.png', dpi=50)
    plt.show()