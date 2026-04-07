from gstatsMCMC.MCMC import *
from gstatsMCMC import Topography
import torch
import cupy as cp
from copy import deepcopy
from . import gstatsim_custom_gpu as gsim

"""
Cupy
- Invoke kernel (e.g. Elementwise, ReductionKernel, RawKernel)

Goals:
1. Minimize CPU to GPU memory transfer
2. Keep RNG on GPU
3. 
"""

def get_mass_conservation_residual_GPU(bed, surf, velx, vely, dhdt, smb, resolution):
    thick = surf - bed
    
    dx = cp.gradient(velx*thick, resolution, axis=1)
    dy = cp.gradient(vely*thick, resolution, axis=0)
    
    res = dx + dy + dhdt - smb
    
    return res


def move_cursor_to_line(line_number):
    """Move cursor to specific line for updating in place"""
    sys.stdout.write(f'\033[{line_number};0H')  # Move to line N, column 0
    sys.stdout.flush()

def clear_line():
    """Clear the current line"""
    sys.stdout.write('\033[2K')
    sys.stdout.flush()


# code adopted from gstatsim_custom by Michael
def _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil):
    """
    Sequential Gaussian Simulation with ordinary or simple kriging using nearest neighbors found in an octant search.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        sim_mask (numpy.ndarray or None): Mask True where to do simulation. Default None will do whole grid.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.

    Returns:
        (out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil)
    """
    
    # get masks and gaussian transform data
    cond_msk = ~cp.isnan(grid)

    #out_grid, nst_trans = gaussian_transformation(grid, cond_msk)
    out_grid = grid.copy()

    if sim_mask is None:
        sim_mask = cp.full(xx.shape, True)

    # get index coordinates and filter with sim_mask
    xx_rows, xx_cols = cp.arange(xx.shape[0]) , cp.arange(xx.shape[1])
    ii, jj = cp.meshgrid(xx_rows, xx_cols, indexing = 'ij')

    filtered_ind_flat =  [ii[sim_mask].flatten(), jj[sim_mask].flatten()]
    inds = cp.array(filtered_ind_flat).T

    vario = deepcopy(variogram)

    # turn scalar variogram parameters into grid
    for key in vario:
        if isinstance(vario[key], numbers.Number):
            vario[key] = cp.full(grid.shape, vario[key])

    # mean of conditioning data for simple kriging
    global_mean = cp.mean(out_grid[cond_msk])

    # make stencil for faster nearest neighbor search
    if stencil is None:
        stencil, _, _ = gsim.neighbors_gpu.make_circle_stencil_gpu_safe(xx[:, 0], radius) # Stencil, [xx_st, yy_st]
    return out_grid, cond_msk, inds, vario, global_mean, stencil

# Referenced from Matthew's code
# def sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, rcond=None, seed=None):

def sgs_gpu(xx, yy, grid, variogram, radius=100e3, num_points=32, ktype='ok', 
            sim_mask=None, quiet=False, stencil=None, seed=None, max_memory_gb=150.0,
            batch_size=None, use_sector_balance=True, n_sectors=8, dtype=cp.float64):
    """
    Sequential Gaussian Simulation (SGS) on GPU.
    OPTIMIZED VERSION.
    
    The key difference from Kriging is that simulated values are added to the 
    conditioning data ('cond_msk') instantly, affecting subsequent points.
    """

    assert isinstance(xx, cp.ndarray), 'Error: xx is not a Cupy Array'
    assert isinstance(yy, cp.ndarray), 'Error: yy is not a Cupy Array'
    assert isinstance(grid, cp.ndarray), 'Error: grid is not a Cupy Array'
    
    gsim._sanity_checks_gpu(xx, yy, grid, variogram, radius, num_points, ktype)
    out_grid, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)

    rng = cp.random.default_rng(seed)

    shuffled = cp.copy(inds)
    cp.random.shuffle(shuffled)

    xx_rows, xx_cols = cp.arange(xx.shape[0]) , cp.arange(xx.shape[1])
    ii, jj = cp.meshgrid(xx_rows, xx_cols, indexing = 'ij')

    ## Batch size optimization | GPU limited VRAM
    if batch_size is None:
        # Approximate VRAM usage per point (bytes):
        # 1. Neighbor Search: Stencil size * bytes (distances)
        # 2. Kriging Matrix: (K+1)^2 * bytes
        # 3. Intermediate tensors (coords, etc): K * 4*bytes
        
        bytes_per = 4 if dtype == cp.float32 else 8 # bytes
        
        # Estimate stencil size (area of circle in pixels)
        dx = float(xx[0,1] - xx[0,0])
        stencil_pixels = int(3.14159 * (radius/dx)**2)
        
        mem_per_point = (stencil_pixels * bytes_per) + (num_points**2 * bytes_per) + (num_points * 128)
        
        # Use 80% of available memory safe margin
        avail_mem = max_memory_gb * (1024**3) * 0.8
        calc_batch = int(avail_mem // mem_per_point)
        
        # Clamp between reasonable limits (e.g., 4k to 200k)
        batch_size = max(4096, min(calc_batch, 200000))
        
        if not quiet:
            print(f"Calculated optimal batch size: {batch_size} (Stencil ~{stencil_pixels} px)")
    
    n_points = shuffled.shape[0]
    pbar = tqdm(total = n_points, desc = "Sequential Gaussian Simulation", unit = "pts")

    for bstart in range(0, n_points, batch_size):
        bend = min(n_points, b_start + batch_size)
        batch_inds = shuffled[bstart:bend].astype(cp.int32)

        # 1. Find neighbors
        neigh, nb_counts = gsim.interpolate_gpu.batch_neighbors_distance_based(
            batch_inds, ii, jj, xx, yy, out_grid, cond_msk, radius, num_points, max_memory_gb, dtype=dtype
        )

        sim_points = cp.stack([xx[batch_inds[:,0], batch_inds[:,1]], 
                               yy[batch_inds[:,0], batch_inds[:,1]]], axis=1)

        valid_mask = counts > 0
        if not cp.any(valid_mask):
            pbar.update(bend - bstart)
            continue
        
        sim_points_valid = sim_points[valid_mask]
        neighbors_valid = neighbors[valid_mask]

        # 2. Kriging
        if ktype == 'ok':
            ests, vars_ = gsim.interpolate_gpu.batch_ok_solve_gpu(sim_points_valid, neighbors_valid, vario, dtype=dtype)
        else:
            ests, vars_ = gsim.interpolate_gpu.batch_sk_solve_gpu(sim_points_valid, neighbors_valid, vario, global_mean, dtype=dtype)

        # 3. Sample from Local Condition Distribution
        vars_safe = cp.abs(vars_)
        std_norm = rng.standard_normal(size=ests.shape, dtype=dtype)
        samp = std_norm * cp.sqrt(vars_safe) + ests

        # 4. Update Grid and Mask

        valid_rows, valid_cols = batch_inds[valid_mask, 0], batch_inds[valid_mask, 1]
        out_grid[valid_rows, valid_cols]

        # Update cond mask
        cond_msk[valid_rows, valid_cols] = True
        pbar.update(bend - bstart)
    
    pbar.close()
    
    out_grid = cp.nan_to_num(out_grid, nan=0.0, posinf=5.0, neginf=-5.0)
    out_grid = cp.clip(out_grid, -5.0, 5.0)

    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1, 1)).squeeze().reshape(xx.shape)

    return sim_trans


class chain_sgs_gpu(chain):
    """
    Inherit the chain class. Used for creating sequential gaussian simulation blocks-based MCMC chains.
    
    Parameters in addition to chain's parameters:
        do_transform (bool): If true, normalize radar measurements with normal score transformation to generate subglacial topography. If false, directly use Sequential Gaussian Simulation on un-normalized subglacial topography.
        nst_trans (scikit-learn.preprocessing.QuantileTransformer): The normal score transformation for the (detrended/not detrended) subglacial topography.
        trend: (numpy.ndarray): A 2D array representing the trend of the subglacial topography
        detrend_map (bool): If 'True', the subglacial topography will be de-trended using parameter 'trend'. If 'False', the topography will not be de-trended.
        vario_type (string): The type of variogram model used for SGS ('Gaussian', 'Exponential', 'Spherical', or 'Matern').
        vario_param (list): A list of parameters defining the variogram model, set by the `set_variogram` method. [azimuth, nugget, major range, minor range, sill, variogram type, smoothness].
        sgs_param (list): A list of parameters controlling the SGS behavior, set by the `set_sgs_param` method. [number of nearest neighbors, searching radius, randomly drop out conditioning data (True or False), dropout rate]
        block_min_x, block_max_x, block_min_y, block_max_y (int): the minimum and maximum width and height of the update block
    """ 
    def __init__(self, xx, yy, initial_bed, surf, velx, vely, dhdt, smb, 
                 cond_bed, data_mask, grounded_ice_mask, resolution):
        # Let the parent do validation and attribute assignment
        super().__init__(xx, yy, initial_bed, surf, velx, vely, dhdt, smb,
                         cond_bed, data_mask, grounded_ice_mask, resolution)
        
        # Convert all 2D grid arrays to CuPy in place | no copy
        self.xx                = cp.asarray(self.xx)
        self.yy                = cp.asarray(self.yy)
        self.initial_bed       = cp.asarray(self.initial_bed)
        self.surf              = cp.asarray(self.surf)
        self.velx              = cp.asarray(self.velx)
        self.vely              = cp.asarray(self.vely)
        self.dhdt              = cp.asarray(self.dhdt)
        self.smb               = cp.asarray(self.smb)
        self.cond_bed          = cp.asarray(self.cond_bed)
        self.data_mask         = cp.asarray(self.data_mask)
        self.grounded_ice_mask = cp.asarray(self.grounded_ice_mask)

        # resolution stays the same (float)
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using the object\'s function set_update_in_region(region_mask) and set_update_region(update_in_region)')
        print('please also set up the sgs parameters using set_sgs_param(self, block_size, sgs_param)')
        print('then please set up the loss function using either set_loss_type or set_loss_func')

    def loss(self, massConvResidual, dataDiff):
        loss_mc = cp.nansum(cp.square(massConvResidual[self.mc_region_mask == 1])) / (2 * self.sigma_mc ** 2)
        loss_data = 0
        return loss_mc + loss_data, loss_mc, loss_data
        
    def set_normal_transformation(self, nst_trans, do_transform = True):
        """
        Set the normal score transformation object (from scikit-learn package) used to normalize the bed elevation.
        The function has no returns. Its effect can be checked in the object's parameter 'nst_trans'
        
        Args:
            nst_trans (QuantileTransformer): A fitted scikit-learn transformer used to normalize input data.
        
        Note:
            This transformation must be fit beforehand (e.g., via `MCMC.fit_variogram`).
        """
        self.do_transform = do_transform
        if do_transform:
            self.nst_trans = nst_trans
        else:
            self.nst_trans = None
      
    def set_trend(self, trend = None, detrend_map = True):
        """
        Set the long-wavelength trend component of the bed topography.
        Notice that detrend topography means that the SGS simulation will only simulate the short-wavelength topography residuals that is not a part of the trend
        The function has no returns. Its effect can be checked in the object's parameter 'trend' and 'detrend_map'
        
        Args:
            trend (np.ndarray): A 2D array, representing the topographic trend.
            detrend_map (bool): If True, remove trend before transforming the bed elevation and add it back after inverse transform.
        
        Raises:
            ValueError: If detrend_map is True but trend has invalid shape.
        """

        assert isinstance(trend, cp.ndarray), 'Error: trend is not a Cupy Array'

        if detrend_map == True:
            if len(trend)!=len(self.xx) or trend.shape != self.xx.shape:
                raise ValueError('if detrend_map is set to True, then the trend of the topography, which is a 2D numpy array, must be provided')
            else:
                self.trend = trend
        else:
            self.trend = None
        self.detrend_map = detrend_map
    
    def set_variogram(self, vario_type, vario_range, vario_sill, vario_nugget, isotropic = True, vario_smoothness = None, vario_azimuth = None):
        """
        Specify variogram model and its parameters for SGS interpolation.
        The function has no returns. Its effect can be checked in the object's parameter 'vario_type' and 'vario_param'
        
        Args:
            vario_type (str): Variogram model type. One of 'Gaussian', 'Exponential', 'Spherical', 'Matern'.
            vario_range (float or list): Correlation range(s). One value for isotropic; list of two for anisotropic.
            vario_sill (float): Variogram sill (variance).
            vario_nugget (float): Nugget effect.
            isotropic (bool): Whether the variogram is isotropic. Default is True.
            vario_smoothness (float): Smoothness parameter for Matern model (required if `vario_type` is 'Matern').
            vario_azimuth (float): Azimuth angle for anisotropic variograms in degrees. Units is degrees (360 maximum)
        
        Raises:
            ValueError: If required parameters are missing or in the wrong format.
        """
            
        if (vario_type == 'Gaussian') or (vario_type == 'Exponential') or (vario_type == 'Spherical'):
            print('the variogram is set to type', vario_type)
        elif vario_type == 'Matern':
            if (vario_smoothness == None) or (vario_smoothness <= 0):
                raise ValueError('vario_smoothness argument should be a positive float when the vario_type is Matern')
            else:
                print('the variogram is set to type', vario_type)
        else:
            raise ValueError('vario_type argument should be one of the following: Gaussian, Exponential, Spherical, or Matern')
        
        self.vario_type = vario_type
        
        if isotropic:
            vario_azimuth = 0
            self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_range, vario_sill, vario_type, vario_smoothness]
        else:
            if (len(vario_range) == 2):
                print('set to anistropic variogram with major range and minor range to be ', vario_range)
                self.vario_param = [vario_azimuth, vario_nugget, vario_range[0], vario_range[1], vario_sill, vario_type, vario_smoothness]
            else:
                raise ValueError ("vario_range need to be a list with two floats to specifying for major range and minor range of the variogram when isotropic is set to False")
    
    def set_sgs_param(self, sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on = False, dropout_rate = 0):
        """
        Set parameters for Sequential Gaussian Simulation (SGS). Details please see implementation of SGS in GStatSim
        The function has no returns. Its effect can be checked in the object's parameter 'sgs_param
        
        Args:
            sgs_num_nearest_neighbors (int): Number of nearest neighbors used in simulation.
            sgs_searching_radius (float): Radius (in meters) to search for neighbors.
            sgs_rand_dropout_on (bool): Whether to randomly drop conditioning points in simulation block.
            dropout_rate (float): Proportion of conditioning data to drop if dropout is enabled (between 0 and 1).
        """
        
        if sgs_rand_dropout_on == False:
            dropout_rate = 0
            print('because the sgs_rand_dropout_on is set to False, the dropout_rate is automatically set to 0')
            
        self.sgs_param = [sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on, dropout_rate]
    
    def set_block_sizes(self, block_min_x, block_max_x, block_min_y, block_max_y):
        """
        Set minimum and maximum block sizes (in grid cells) for SGS updates.
        The function has no returns. Its effect can be checked in the object's parameter 'block_min_x', 'block_max_x', 'block_min_y', 'block_max_y'
        
        Args:
            block_min_x (int): Minimum width of block in x-direction. Unit in grid cells
            block_max_x (int): Maximum width of block in x-direction.
            block_min_y (int): Minimum height of block in y-direction.
            block_max_y (int): Maximum height of block in y-direction.
        """
        self.block_min_x = block_min_x
        self.block_min_y = block_min_y
        self.block_max_x = block_max_x
        self.block_max_y = block_max_y
    
    def set_random_generator(self, rng_seed = None):
        """
        Set the random generator for the chain to maintain replicability
        Notice that once set_random_generator is called, the random generator for all following call of "run()" will use the continuous sequence of randomness defined by the random generator here. In other words, no other object of random generator will be created unless the set_random_generator() is called again.
        Default: to run a complete chain, call set_random_generator only once before the chain started.

        Args:
            rng_seed (None, int, or numpy.random._generator.Generator): set the random number generator for the chain, either by assigning a random default rng (rng_seed = None), a random generator with seed of rng_seed (rng_seed has type int), or a random generator defined by rng_seed (rng_seed is a generator itself).
        """
        if rng_seed is None:
            rng = cp.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = cp.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, cp.random._generator.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a CuPy random Generator, or None')
            
        self.rng = rng

    def run_all_assertions(self):
        assert isinstance(self.xx, cp.ndarray),                'Error: xx is not a CuPy array'
        assert isinstance(self.yy, cp.ndarray),                'Error: yy is not a CuPy array'
        assert isinstance(self.initial_bed, cp.ndarray),       'Error: initial_bed is not a CuPy array'
        assert isinstance(self.surf, cp.ndarray),              'Error: surf is not a CuPy array'
        assert isinstance(self.velx, cp.ndarray),              'Error: velx is not a CuPy array'
        assert isinstance(self.vely, cp.ndarray),              'Error: vely is not a CuPy array'
        assert isinstance(self.dhdt, cp.ndarray),              'Error: dhdt is not a CuPy array'
        assert isinstance(self.smb, cp.ndarray),               'Error: smb is not a CuPy array'
        assert isinstance(self.cond_bed, cp.ndarray),          'Error: cond_bed is not a CuPy array'
        assert isinstance(self.data_mask, cp.ndarray),         'Error: data_mask is not a CuPy array'
        assert isinstance(self.grounded_ice_mask, cp.ndarray), 'Error: grounded_ice_mask is not a CuPy array'
        assert isinstance(self.region_mask, cp.ndarray),       'Error: region_mask is not a CuPy array. Did you call set_update_region()?'
        assert isinstance(self.mc_region_mask, cp.ndarray),    'Error: mc_region_mask is not a CuPy array. Did you call set_loss_type()?'

        # Trend is only required when detrending is enabled
        if self.detrend_map:
            assert isinstance(self.trend, cp.ndarray),         'Error: trend is not a CuPy array. Did you call set_trend()?'

        # ── Scalar / non-array parameter checks ────────────────────────
        assert isinstance(self.resolution, (int, float)),      'Error: resolution is not a scalar'
        assert isinstance(self.sigma_mc, (int, float)),        'Error: sigma_mc is not set. Did you call set_loss_type()?'
        assert self.sigma_mc > 0,                              'Error: sigma_mc must be positive'
        assert hasattr(self, 'vario_param'),                   'Error: vario_param is not set. Did you call set_variogram()?'
        assert hasattr(self, 'sgs_param'),                     'Error: sgs_param is not set. Did you call set_sgs_param()?'
        assert hasattr(self, 'block_min_x'),                   'Error: block sizes not set. Did you call set_block_sizes()?'

        # ── RNG check ──────────────────────────────────────────────────
        assert hasattr(self, 'rng'),                           'Error: rng is not set. Did you call set_random_generator()?'

        # ── Optional: sample_loc check ─────────────────────────────────
        if self.sample_loc is not None:
            assert isinstance(self.sample_loc, cp.ndarray),    'Error: sample_loc is not a CuPy array'

    def run(self, n_iter, only_save_last_bed=False, info_per_iter=100, plot=True, progress_bar=True):
        """
        Run the MCMC chain using block-based SGS updates, with the new gstatsim_custom code
        
        Args:
            n_iter (int): Number of iterations in the MCMC chain.
            only_save_last_bed: If true, the function will only return one subglacial topography at the end of iterations. If false, the function will return all subglacial topography in every iteration.
            info_per_iter (int): for every this number of iterations, the information regarding current loss values and acceptance rate will be printed out.
        
        Returns:
            bed_cache (np.ndarray): A 3D array showing subglacial topography at each iteration, or only the last topography.
            loss_mc_cache (np.ndarray): A 1D array of mass conservation loss at each iteration. If the mass conservation loss is not used, return array of 0
            loss_data_cache (np.ndarray): A 1D array of data misfit loss at each iteration. If the data misfit loss is not used, return array of 0
            loss_cache (np.ndarray): A 1D array of total loss at each iteration.
            step_cache (np.ndarray): A 1D array of boolean indicating if the step was accepted.
            resampled_times (np.ndarray): A 2D array of number of times each pixel was updated.
            blocks_cache (np.ndarray): A 1D array of info on block proposals at each iteration, (x coordinate for the center of the block, y coordinate for the center of the block, block size in x-direction, block size in y-direction).
        """
            
        rng = self.rng
        run_all_assertions()
        
        rows = self.xx.shape[0]
        cols = self.xx.shape[1]
        
        loss_cache      = cp.zeros(n_iter)
        loss_mc_cache   = cp.zeros(n_iter)
        loss_data_cache = cp.zeros(n_iter)
        step_cache      = cp.zeros(n_iter)

        if not only_save_last_bed:
            bed_cache   = cp.zeros((n_iter, rows, cols))

        blocks_cache    = cp.full((n_iter, 4), np.nan)
        resampled_times = cp.zeros(self.xx.shape)
        
        # if the user request to return bed elevation in some sampling locations
        if not (self.sample_loc is None):
            sample_values = cp.zeros((self.sample_loc.shape[0], n_iter))
            
            # convert sample_loc from x and y locations to i and j indexes
            sample_loc_ij = cp.zeros(self.sample_loc.shape, dtype=np.int16)
            for k in range(self.sample_loc.shape[0]):
                sample_i,sample_j = cp.where((self.xx == self.sample_loc[k,0]) & (self.yy == self.sample_loc[k,1]))
                sample_loc_ij[k,:] = [int(sample_i[0]), int(sample_j[0])]
                
            sample_values[:,0] = self.initial_bed[sample_loc_ij[:,0],sample_loc_ij[:,1]]

        if self.detrend_map:
            bed_c = (self.initial_bed - self.trend).copy()
            cond_bed_c = (self.cond_bed - self.trend).copy()
        else:
            bed_c = self.initial_bed.copy()
            cond_bed_c = self.cond_bed.copy()
       
        if self.do_transform:
            nst_trans = self.nst_trans
            z = nst_trans.transform(bed_c.reshape(-1,1))
            z_cond_bed = nst_trans.transform(cond_bed_c.reshape(-1,1))
        else:
            z = bed_c.copy().reshape(-1,1)
            z_cond_bed = cond_bed_c.copy().reshape(-1,1)
    
        z_cond_bed = z_cond_bed.reshape(self.xx.shape)

        resolution = self.resolution
        
        # initialize loss
        if self.detrend_map == True:
            mc_res = Topography.get_mass_conservation_residual(bed_c + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        else:
            mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        
        data_diff = bed_c - cond_bed_c
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)
    
        loss_cache[0] = loss_prev
        loss_mc_cache[0] = loss_prev_mc
        loss_data_cache[0] = loss_prev_data
        step_cache[0] = False
        if not only_save_last_bed:
            bed_cache[0] = bed_c
    
        rad = self.sgs_param[1]
        neighbors = self.sgs_param[0]
        
        if self.vario_param[5] == 'Matern':
            vario = {
                'azimuth' : self.vario_param[0],
                'nugget' : self.vario_param[1],
                'major_range' : self.vario_param[2],
                'minor_range' : self.vario_param[3],
                'sill' :  self.vario_param[4],
                'vtype' : self.vario_param[5],
                's' : self.vario_param[6]
            }
        else:
            vario = {
                'azimuth' : self.vario_param[0],
                'nugget' : self.vario_param[1],
                'major_range' : self.vario_param[2],
                'minor_range' : self.vario_param[3],
                'sill' :  self.vario_param[4],
                'vtype' : self.vario_param[5],
            }

        # plotting for real-time result update
        if plot:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12,5))
            (line_loss,) = ax_loss.plot([], [], color='tab:blue', label='Loss')
            (line_acc,)  = ax_acc.plot([], [], color='tab:green', label='Acceptance Rate')
            #NOTE use get_mass_conservation_residual on BedMachine data
            # bm_loss = 
            # ax_loss.axhline(bm_loss, ls='--', label='BedMachine loss') 
            
            ax_loss.set_xlabel("Iteration")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("MCMC Loss")

            ax_acc.set_xlabel("Iteration")
            ax_acc.set_ylabel("Acceptance Rate (%)")
            ax_acc.set_ylim(0, 100)
            ax_acc.set_title("MCMC Acceptance Rate")

            ax_loss.legend()
            ax_acc.legend()
            
            display_handle = display.display(fig, display_id=True)
            plt.tight_layout()

        # Track acceptance rate
        accepted_count = 0
        acceptance_rates = []

        if progress_bar == True:
            chain_id = getattr(self, 'chain_id', 0)
            seed = getattr(self, 'seed', 'Unknown')
            tqdm_position = getattr(self, 'tqdm_position', 0)

            iterator = tqdm(range(1,n_iter),
                            desc=f'Chain {chain_id} | Seed {seed}',
                            position=tqdm_position,
                            leave=True) 
        else:
            iterator = range(1,n_iter)

            chain_id = getattr(self, 'chain_id', 'Unknown')
            output_line = getattr(self, 'tqdm_position', 0) + 2 # Reserve first line for header
            seed = getattr(self, 'seed', 'Unknown')

        iter_start_time = time.time()        
        for i in range(n_iter):
    
            while True:
                indexx = rng.integers(low=0, high=bed_c.shape[0], size=1)[0]
                indexy = rng.integers(low=0, high=bed_c.shape[1], size=1)[0]
                if self.region_mask[indexx,indexy] == 1:
                    break
    
            block_size_x = rng.integers(low=self.block_min_x, high=self.block_max_x, size=1)[0]
            block_size_y = rng.integers(low=self.block_min_y, high=self.block_max_y, size=1)[0]
    
            blocks_cache[i,:]=[indexx,indexy,block_size_x,block_size_y]
    
            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = cp.max((0,int(indexx-block_size_x/2)))
            bxmax = cp.min((bed_c.shape[0],int(indexx+block_size_x/2)))
            bymin = cp.max((0,int(indexy-block_size_y/2)))
            bymax = cp.min((bed_c.shape[1],int(indexy+block_size_y/2)))
    
            if self.do_transform == True:
                bed_tosim = nst_trans.transform(bed_c.reshape(-1,1)).reshape(self.xx.shape)
            else:
                bed_tosim = bed_c.copy()
    
            bed_tosim[bxmin:bxmax,bymin:bymax] = z_cond_bed[bxmin:bxmax,bymin:bymax].copy()
            sim_mask = np.full(self.xx.shape, False)
            sim_mask[bxmin:bxmax,bymin:bymax] = True
            newsim = sgs(self.xx, self.yy, bed_tosim, vario, rad, neighbors, sim_mask = sim_mask, seed=rng)
    
            if self.do_transform == True:
                bed_next = nst_trans.inverse_transform(newsim.reshape(-1,1)).reshape(rows,cols)
            else:
                bed_next = newsim.copy()
            
            if self.detrend_map == True:
                #mc_res = Topography.get_mass_conservation_residual(bed_next + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
                mc_res = get_mass_conservation_residual_GPU(bed_next + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)

            else:
                mc_res = get_mass_conservation_residual_GPU(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
            
            data_diff = bed_next - cond_bed_c
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
            
            if self.detrend_map == True:
                block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - (bed_next[bxmin:bxmax,bymin:bymax] + self.trend[bxmin:bxmax,bymin:bymax])
            else:
                block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - bed_next[bxmin:bxmax,bymin:bymax]
                
            block_region_mask = self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]
                
            if cp.sum((block_thickness<=0)&(block_region_mask==1)) > 0:
                loss_next = cp.inf

            if loss_prev > loss_next:
                acceptance_rate = 1
                
            else:
                acceptance_rate = min(1 , cp.exp(loss_prev-loss_next))
    
            u = rng.random()
            
            if (u <= acceptance_rate):
                bed_c               = bed_next
                loss_cache[i]       = loss_next
                loss_mc_cache[i]    = loss_next_mc
                loss_data_cache[i]  = loss_next_data
                step_cache[i]       = True
                
                loss_prev       = loss_next
                loss_prev_mc    = loss_next_mc
                loss_prev_data  = loss_next_data
                resampled_times[bxmin:bxmax,bymin:bymax] += 1
            else:
                loss_cache[i]       = loss_prev
                loss_mc_cache[i]    = loss_prev_mc
                loss_data_cache[i]  = loss_prev_data
                step_cache[i]       = False
    
            if not only_save_last_bed:
                if self.detrend_map == True:
                    bed_cache[i,:,:] = bed_c + self.trend
                else:
                    bed_cache[i,:,:] = bed_c
                    
            if not (self.sample_loc is None):
                sample_values[:,i] = bed_c[sample_loc_ij[:,0],sample_loc_ij[:,1]]

            if progress_bar:
                # Update tqdm progress bar
                iterator.set_postfix({
                    'chain_id'  :   chain_id,
                    'seed'      :   seed,
                    'mc loss'   :   f'{loss_mc_cache[i]:.3e}',
                    'data loss' :   f'{loss_data_cache[i]:.3e}',
                    'loss'      :   f'{loss_cache[i]:.3e}',
                    'acceptance rate'   :   f'{np.sum(step_cache)/(i+1):.6f}'
                })
            else:
                if i%info_per_iter == 0 or i == 1 or i == n_iter - 1:
                    move_cursor_to_line(output_line)
                    clear_line()
                    
                    # Calculate progress
                    progress = i / (n_iter - 1)
                    progress_pct = progress * 100
                    elapsed = time.time() - iter_start_time
                    iter_per_sec = i / elapsed if elapsed > 0 else 0
                    
                    # Calculate ETA
                    if iter_per_sec > 0:
                        remaining_iters = n_iter - i
                        eta_seconds = remaining_iters / iter_per_sec
                        eta_hours = int(eta_seconds // 3600)
                        eta_minutes = int((eta_seconds % 3600) // 60)
                        eta_secs = int(eta_seconds % 60)
                        eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
                    else:
                        eta_str = "--:--:--"
                    
                    # Create visual progress bar
                    bar_length = 10
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '▍' * (1 if filled_length < bar_length and progress > 0 else 0)
                    bar = bar.ljust(bar_length)
                    
                    # Format output
                    print(f'Chain {chain_id} ({str(seed)[:6]}): {progress_pct:3.0f}%|{bar}| ETA: {eta_str} | it/s: {iter_per_sec:6.2f} | n: {n_iter:{len(str(n_iter))}d} | loss: {loss_cache[i]:.3e} | acc: {np.sum(step_cache)/(i+1):.4f}', end='')
                    sys.stdout.flush()

            # Calculate acceptance rate for plot
            total_acceptance = (accepted_count / (i + 1)) * 100
            acceptance_rates.append(total_acceptance)

            if plot:
                if i < 5000:
                    update_interval = 100
                else:
                    update_interval = info_per_iter

                if i % update_interval == 0:
                    # Update loss line
                    line_loss.set_data(range(i + 1), loss_cache[:i + 1])
                    ax_loss.relim()
                    ax_loss.autoscale_view()

                    # Update acceptance rate line
                    line_acc.set_data(range(len(acceptance_rates)), acceptance_rates)
                    ax_acc.set_ylim(0, 100)
                    ax_acc.relim()
                    ax_acc.autoscale_view()

                    display_handle.update(fig)
    
        if self.detrend_map == True:
            last_bed = bed_c + self.trend
        else:
            last_bed = bed_c
    
        if not only_save_last_bed:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
        else:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return last_bed, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache


