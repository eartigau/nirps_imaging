"""
Guiding RMS Analysis Tool
=====================

Analyzes guiding image data from FITS files to extract RMS properties and radial profiles.
Extracts the 'GUIDING' extension, computes the optimal center position, generates radial 
and angular profiles, and visualizes residuals.

Usage:
    python get_props_guiding.py <filepath>

Author: Étienne Artigau
Date: 2026-03-27
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from astropy.wcs import WCS
import warnings
from astropy.io.fits.verify import VerifyWarning
import os
import glob
import time
import yaml
import getpass
import fnmatch
import io
from contextlib import redirect_stdout

# ---------------------------------------------------------------------------
# Load configuration from YAML (falls back to built-in defaults if not found)
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'guiding_config.yaml')
_DEFAULTS = {
    'xcen_default': 360,
    'ycen_default': 220,
    'rad_rms': 50,
    'rad_rms_trim': 5,
    'rad_bin_step_inner': 0.5,
    'rad_bin_inner_max': 5,
    'rad_bin_step_outer_factor': 0.1,
    'angular_bin_size': 30,
    'angular_fit_harmonics': 5,
    'angular_fit_step_deg': 0.1,
    'robust_mean_sigma': 5,
    'hatch_remove_sigma': 3,
    'wcs_cdelt': 0.1,
    'wcs_rotation_angle': 45.0,
    'plot_residual_scale': 0.1,
}

def load_config(config_path=_CONFIG_PATH):
    """Load YAML config, merging defaults and resolving user-specific folders."""
    cfg = dict(_DEFAULTS)
    cfg['data_folder'] = ''
    cfg['output_folder'] = ''

    if os.path.exists(config_path):
        with open(config_path, 'r') as fh:
            loaded = yaml.safe_load(fh) or {}

        # Apply non-folder keys directly; folder paths are resolved from user map.
        for key, value in loaded.items():
            if key != 'user':
                cfg[key] = value

        users_cfg = loaded.get('user', {})
        if not isinstance(users_cfg, dict):
            print("Warning: 'user' section must be a mapping. Falling back to empty folder defaults.")
            users_cfg = {}

        default_user_cfg = users_cfg.get('default', {})
        if default_user_cfg is None:
            default_user_cfg = {}
        if not isinstance(default_user_cfg, dict):
            print("Warning: user.default must be a mapping. Ignoring it.")
            default_user_cfg = {}

        current_user = getpass.getuser()
        selected_user_cfg = users_cfg.get(current_user)
        if selected_user_cfg is None:
            # No exact match — try wildcard patterns (skip reserved keys)
            reserved = {'default'}
            matched_key = None
            for key in users_cfg:
                if key not in reserved and fnmatch.fnmatch(current_user, key):
                    matched_key = key
                    break
            if matched_key is not None:
                print(
                    f"Info: user '{current_user}' matched wildcard pattern '{matched_key}' "
                    "in config 'user' section."
                )
                selected_user_cfg = users_cfg[matched_key]
            else:
                print(
                    f"Warning: user '{current_user}' not found in config 'user' section. "
                    "Falling back to user.default."
                )
                selected_user_cfg = {}
        if not isinstance(selected_user_cfg, dict):
            print(
                f"Warning: matched user entry for '{current_user}' must be a mapping. "
                "Falling back to user.default."
            )
            selected_user_cfg = {}

        merged_user_cfg = dict(default_user_cfg)
        merged_user_cfg.update(selected_user_cfg)
        cfg['data_folder'] = merged_user_cfg.get('data_folder', '')
        cfg['output_folder'] = merged_user_cfg.get('output_folder', '')
        cfg['wildcard'] = merged_user_cfg.get('wildcard', '')
    else:
        print(f"Warning: Config file not found at {config_path}. Using built-in defaults.")
    return cfg

CONFIG = load_config()


def get_output_path(filepath, output_folder=None):
    """Build the expected output product path for an input FITS file."""

    if output_folder is not None:
        output_filename = os.path.basename(filepath).replace('.fits', '_guiding_analysis.fits')
        return os.path.join(output_folder, output_filename)
    return filepath.replace('.fits', '_guiding_analysis.fits')


def get_skip_token_path(filepath, output_folder=None):
    """Build hidden token path used to mark files with no guiding window."""

    token_name = f".{os.path.basename(filepath)}"
    if output_folder is not None:
        return os.path.join(output_folder, token_name)
    return os.path.join(os.path.dirname(filepath), token_name)


def touch_skip_token(filepath, output_folder=None):
    """Create an empty hidden token file to skip this input on future runs."""

    token_path = get_skip_token_path(filepath, output_folder=output_folder)
    token_dir = os.path.dirname(token_path) or '.'
    os.makedirs(token_dir, exist_ok=True)
    with open(token_path, 'a', encoding='utf-8'):
        pass
    os.utime(token_path, None)
    return token_path


def format_duration(total_seconds):
    """Format duration as days/hours/minutes/seconds."""

    seconds = max(0, int(round(float(total_seconds))))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {secs}s"

def smart_fmt(value):
    """Format floats with <=2 decimals when |x|>1, otherwise <=5 decimals."""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(v):
        return str(v)

    if abs(v) > 1:
        return f"{v:.2f}".rstrip('0').rstrip('.')
    return f"{v:.5f}".rstrip('0').rstrip('.')

def robust_mean(x):
    """Compute robust mean resistant to outliers."""

    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    if not np.isfinite(mad) or mad == 0:
        return float(median)

    inliers = np.abs(values - median) < CONFIG['robust_mean_sigma'] * mad
    if not np.any(inliers):
        return float(median)

    return float(np.nanmean(values[inliers]))

def filter_profile_points(rad_profile, profile):
    """Keep only finite radial-profile points for interpolation."""

    keep = np.isfinite(rad_profile) & np.isfinite(profile)
    return rad_profile[keep], profile[keep]

def fit_angular_harmonics(theta_deg, values, max_harmonics=5, step_deg=0.1):
    """Fit a Fourier harmonic series (cos+sin) and sample it finely."""

    theta_deg = np.asarray(theta_deg, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(theta_deg) & np.isfinite(values)
    theta_deg = theta_deg[keep]
    values = values[keep]

    if theta_deg.size < 2:
        return None

    # Two parameters per harmonic (cos and sin) plus constant term.
    # Keep the system overdetermined to avoid unstable fits.
    n_harmonics = max(1, min(int(max_harmonics), (theta_deg.size - 1) // 2))
    theta_rad = np.radians(theta_deg)
    design = [np.ones_like(theta_rad)]
    for order in range(1, n_harmonics + 1):
        design.append(np.cos(order * theta_rad))
        design.append(np.sin(order * theta_rad))
    design = np.column_stack(design)

    coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)

    n_eval = max(360, int(np.ceil(360.0 / step_deg)))
    theta_fine = np.linspace(0.0, 360.0, n_eval, endpoint=False)
    theta_fine_rad = np.radians(theta_fine)
    fit_values = np.full_like(theta_fine, coeffs[0], dtype=float)
    coeff_idx = 1
    for order in range(1, n_harmonics + 1):
        fit_values += coeffs[coeff_idx] * np.cos(order * theta_fine_rad)
        coeff_idx += 1
        fit_values += coeffs[coeff_idx] * np.sin(order * theta_fine_rad)
        coeff_idx += 1

    peak_idx = int(np.nanargmax(fit_values))
    return {
        'theta_deg': theta_fine,
        'fit_values': fit_values,
        'peak_angle_deg': float(theta_fine[peak_idx]),
        'peak_value': float(fit_values[peak_idx]),
        'coeffs': coeffs,
        'n_harmonics': n_harmonics,
    }

def remove_hatch(image):
    mask = np.ones_like(image, dtype=float)


    for ite in range(2):
        for col in range(image.shape[1]):
            col_data = image[:, col]+mask[:, col]
            if np.any(~np.isnan(col_data)):
                image[:, col] -= robust_mean(col_data)
        for row in range(image.shape[0]):
            row_data = image[row, :]+mask[row, :]
            if np.any(~np.isnan(row_data)):
                image[row, :] -= robust_mean(row_data)

        mask = np.ones_like(image, dtype=float)
        threshold = CONFIG['hatch_remove_sigma'] * np.nanmedian(image)
        mask[np.isfinite(image) & (image > threshold)] = np.nan

    return image

def analyze_guiding_image(
    filepath,
    doplot=False,
    output_folder=None,
    force=False,
    save_figures=None,
    save_figures_basename=None,
):
    """
    Analyze guiding image data from a FITS file.
    
    Parameters
    ----------
    filepath : str
        Path to the FITS file containing guiding image data
    doplot : bool, optional
        If True, display matplotlib plots interactively. Default is False.
    output_folder : str, optional
        Folder to save output FITS file. If None, save in same directory as input file.
    force : bool, optional
        If True, reprocess even if the output FITS file already exists.
    save_figures : str, optional
        Folder to save PNG figures. If None, figures are not saved to disk.
    save_figures_basename : str, optional
        Basename used for saved PNG figure names. If None, uses input FITS stem.
        
    Returns
    -------
    str
        Processing status: processed or a skip/error code.
        
    Notes
    -----
    Gracefully skips processing if the 'GUIDING' extension is not found.
    """
    
    # Determine output and hidden-token paths
    output_path = get_output_path(filepath, output_folder=output_folder)
    token_path = get_skip_token_path(filepath, output_folder=output_folder)

    # Skip quickly if a hidden token exists (unless force=True)
    if os.path.exists(token_path) and not force:
        print(f"Skip token already exists: {token_path}. Skipping (use --force to reprocess).")
        return 'skipped_token'
    
    # Skip if output file already exists (unless force=True)
    if os.path.exists(output_path) and not force:
        print(f"Output file already exists: {output_path}. Skipping (use --force to reprocess).")
        return 'skipped_output'
    
    # Attempt to load the GUIDING extension from the FITS file
    try:
        guiding_image0 = fits.getdata(filepath, 'GUIDING').astype(float)
        # Read header from primary HDU for RA/DEC
        hdr = fits.getheader(filepath, 0)
    except KeyError:
        token_path = touch_skip_token(filepath, output_folder=output_folder)
        print(f"Warning: 'GUIDING' extension not found in {filepath}. Created skip token: {token_path}")
        return 'skipped_missing_guiding'
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return 'error'
    
    guiding_image = guiding_image0.copy()
    

    guiding_image = remove_hatch(guiding_image)

    # Subtract median to center the data distribution
    guiding_image -= np.nanmedian(guiding_image)
    
    # Get image dimensions
    sz = guiding_image.shape
    
    # Create coordinate grids for the image (in pixels)
    ypix, xpix = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), indexing='ij')
    ypix, xpix = ypix.astype(float), xpix.astype(float)
    
    # Initial center comes from config and is used as optimizer start point.
    xcen = float(CONFIG.get('xcen_default', 360.0))
    ycen = float(CONFIG.get('ycen_default', 220.0))
    if not (0 <= xcen < sz[1] and 0 <= ycen < sz[0]):
        print(
            f"Warning: Default center ({xcen}, {ycen}) is outside image bounds "
            f"for {filepath}. Skipping analysis."
        )
        return 'error'

    # rad_rms
    rad_rms0 = CONFIG['rad_rms']
        
    def get_profile(xcen, ycen, rad_rms=50):
        """
        Compute 1D radial profile around a center position.
        
        Parameters
        ----------
        xcen : float
            X-coordinate of the center
        ycen : float
            Y-coordinate of the center
            
        Returns
        -------
        rad_profile : ndarray
            Radial bin centers
        profile : ndarray
            Median flux values in each radial bin
        """
        # Compute radial distance from center for all pixels (once)
        rad = np.sqrt((xpix - xcen)**2 + (ypix - ycen)**2)
        
        # Build radial bins: inner step up to rad_bin_inner_max, then fractional step
        rad_profile = [1.0]
        while rad_profile[-1] < rad_rms:
            if rad_profile[-1] <= CONFIG['rad_bin_inner_max']:
                next_bin = CONFIG['rad_bin_step_inner']
            else:
                next_bin = rad_profile[-1] * CONFIG['rad_bin_step_outer_factor']
            rad_profile.append(rad_profile[-1] + next_bin)
        
        rad_profile = np.array(rad_profile)
        
        # Extract flux using vectorized binning
        profile = []
        for i in range(len(rad_profile)):
            r = rad_profile[i]
            if i == 0:
                step = rad_profile[1] - rad_profile[0]
            elif i == len(rad_profile) - 1:
                step = rad_profile[-1] - rad_profile[-2]
            else:
                step = rad_profile[i + 1] - rad_profile[i]
            
            keep = (rad >= (r - step / 2)) & (rad < (r + step / 2))
            if np.any(keep):
                profile.append(robust_mean(guiding_image[keep]))
            else:
                profile.append(np.nan)
        
        profile = np.array(profile)
        
        return rad_profile, profile
    
    def get2dprofile(xcen, ycen, rad_rms=50):
        """
        Create a 2D model profile using spline interpolation of radial profile.
        
        Parameters
        ----------
        xcen : float
            X-coordinate of the center
        ycen : float
            Y-coordinate of the center
            
        Returns
        -------
        image2 : ndarray
            2D model image with NaN outside the fit radius
        """
        # Compute radial distance from center
        rad = np.sqrt((xpix - xcen)**2 + (ypix - ycen)**2)
        
        # Get 1D radial profile
        rad_profile, profile = get_profile(xcen, ycen, rad_rms=rad_rms)
        
        rad_profile, profile = filter_profile_points(rad_profile, profile)
        if rad_profile.size < 2:
            return np.full_like(guiding_image, np.nan, dtype=float)

        # Create linear spline interpolation (faster than cubic for this use case)
        spl = ius(rad_profile, profile, k=1, ext=1)
        
        # Apply spline only to pixels within radius
        keep = rad < rad_rms
        image2 = np.full_like(guiding_image, np.nan, dtype=float)
        image2[keep] = spl(rad[keep])
        
        return image2
    
    def rms2rad(xcen, ycen, rad_rms=50):
        """
        Compute normalized RMS of residuals between data and model.
        
        Parameters
        ----------
        xcen : float
            X-coordinate of the center
        ycen : float
            Y-coordinate of the center
            
        Returns
        -------
        rms : float
            Normalized RMS value
        """
        # Generate 2D model profile
        image2 = get2dprofile(xcen, ycen, rad_rms=rad_rms)
        
        # Compute normalization factor
        norm = np.nansum(image2)
        
        # Compute normalized RMS of residuals
        if norm <= 0 or not np.isfinite(norm):
            return np.inf

        rms = np.nanstd(guiding_image - image2) / norm
        
        return rms
    
    # Minimize RMS to find the optimal center position
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
        res = minimize(
            lambda x: rms2rad(x[0], x[1], rad_rms=rad_rms0),
            [xcen, ycen],
            method='Nelder-Mead'
        )
    xcen_opt, ycen_opt = res.x
    
    rad_rms_trim = CONFIG['rad_rms_trim']
    rad_rms_max = np.nanmax(np.sqrt((xpix - xcen_opt)**2 + (ypix - ycen_opt)**2))
    
    # Get radial profile at optimal center (computed once)
    rad_profile, profile = get_profile(xcen_opt, ycen_opt, rad_rms=rad_rms0)
    
    # Plot 1: Radial profile
    if doplot or save_figures:
        fig_rad, ax_rad = plt.subplots(figsize=(8, 5))
        ax_rad.plot(rad_profile, profile, marker='o')
        ax_rad.set_xlabel('radius (pixels)')
        ax_rad.set_ylabel('median flux')
        ax_rad.set_title('Radial Profile')
        if save_figures:
            os.makedirs(save_figures, exist_ok=True)
            stem = save_figures_basename or os.path.basename(filepath).replace('.fits', '')
            fig_rad.savefig(os.path.join(save_figures, f'{stem}_radial_profile.png'), dpi=150, bbox_inches='tight')
            print(f"Saved: {os.path.join(save_figures, stem + '_radial_profile.png')}")
        if doplot:
            plt.show()
        plt.close(fig_rad)
    
    # Generate 2D model at optimal center
    rad_profile_full, profile_full = get_profile(xcen_opt, ycen_opt, rad_rms=int(rad_rms_max - rad_rms_trim))
    rad_profile_full, profile_full = filter_profile_points(rad_profile_full, profile_full)
    if rad_profile_full.size < 2:
        print(f"Warning: Insufficient valid annuli to build a spline model for {filepath}. Skipping analysis.")
        return 'error'

    spl = ius(rad_profile_full, profile_full, k=1, ext=1)
    
    # Compute radial distance and angular position arrays
    rad = np.sqrt((xpix - xcen_opt)**2 + (ypix - ycen_opt)**2)
    image2 = np.full_like(guiding_image, np.nan, dtype=float)
    keep = rad < int(rad_rms_max - rad_rms_trim)
    image2[keep] = spl(rad[keep])
    
    theta = np.arctan2(ypix - ycen_opt, xpix - xcen_opt) / np.pi * 180 + 180
    
    # Bin angle into angular sectors
    ang_bin = CONFIG['angular_bin_size']
    theta = ang_bin * (theta // ang_bin).astype(int)
    
    # Get unique angle bins and compute residual flux in each bin
    theta_bin = np.unique(theta)
    residual_flux_bin = np.array([
        np.nansum((guiding_image - image2)[(theta == t) & (rad < rad_rms0)])
        for t in theta_bin
    ])  # rad_rms0 from config
    theta_bin_center = (theta_bin + 0.5 * ang_bin) % 360.0
    
    # Extract region of interest around optimal center
    guiding_image_box = guiding_image[
        int(ycen_opt) - rad_rms0:int(ycen_opt) + rad_rms0 + 1,
        int(xcen_opt) - rad_rms0:int(xcen_opt) + rad_rms0 + 1
    ].copy()
    image2_box = image2[
        int(ycen_opt) - rad_rms0:int(ycen_opt) + rad_rms0 + 1,
        int(xcen_opt) - rad_rms0:int(xcen_opt) + rad_rms0 + 1
    ].copy()
    
    # Compute RMS residual before normalization
    residual_unnorm = guiding_image_box - image2_box
    rms_residual = np.nanstd(residual_unnorm)
    
    # Normalize to flux
    norm = np.nansum(image2_box)
    if norm > 0:
        guiding_image_box /= norm
        image2_box /= norm
        residual_flux_bin /= norm
    else:
        print("Warning: Normalization value is zero or negative.")
        return 'error'

    harmonic_fit = fit_angular_harmonics(
        theta_bin_center,
        residual_flux_bin,
        max_harmonics=CONFIG['angular_fit_harmonics'],
        step_deg=CONFIG['angular_fit_step_deg'],
    )
    if harmonic_fit is None:
        print(f"Warning: Could not fit angular harmonics for {filepath}. Skipping analysis.")
        return 'error'
    
    # Plot 2: Angular residuals
    if doplot or save_figures:
        fig_ang, ax_ang = plt.subplots(figsize=(8, 5))
        ax_ang.plot(theta_bin_center, residual_flux_bin, marker='o', linestyle='none', label='Binned residuals')
        ax_ang.plot(
            harmonic_fit['theta_deg'],
            harmonic_fit['fit_values'],
            linewidth=2,
            label=f"Fourier fit ({harmonic_fit['n_harmonics']} harmonics)",
        )
        ax_ang.axvline(
            harmonic_fit['peak_angle_deg'],
            color='tab:red',
            linestyle='--',
            linewidth=1,
            label=(
                f"Peak: {harmonic_fit['peak_angle_deg']:.2f} deg, "
                f"{harmonic_fit['peak_value']:.4f}"
            ),
        )
        ax_ang.set_xlabel('theta (degrees)')
        ax_ang.set_ylabel('residual flux (fraction of total)')
        ax_ang.set_title('Angular Distribution of Residuals')
        ax_ang.legend()
        if save_figures:
            os.makedirs(save_figures, exist_ok=True)
            stem = save_figures_basename or os.path.basename(filepath).replace('.fits', '')
            fig_ang.savefig(os.path.join(save_figures, f'{stem}_angular_residuals.png'), dpi=150, bbox_inches='tight')
            print(f"Saved: {os.path.join(save_figures, stem + '_angular_residuals.png')}")
        if doplot:
            plt.show()
        plt.close(fig_ang)
    
    # Plot 3: Data, model, and residuals comparison
    fig, ax = plt.subplots(
        1, 3,
        figsize=(15, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    
    vmax = np.nanmax(guiding_image_box)
    res_scale = CONFIG['plot_residual_scale']

    # Data image
    ax[0].imshow(guiding_image_box, origin='lower', vmin=0, vmax=vmax)
    ax[0].set_title('Original Data')
    
    # Model image
    ax[1].imshow(image2_box, origin='lower', vmin=0, vmax=vmax)
    ax[1].set_title('Radially-Symmetric Model')
    
    # Residual image
    ax[2].imshow(
        guiding_image_box - image2_box,
        origin='lower',
        vmin=-res_scale * vmax,
        vmax=res_scale * vmax
    )
    ax[2].set_title('Residuals (data − model)')

    if save_figures:
        os.makedirs(save_figures, exist_ok=True)
        stem = save_figures_basename or os.path.basename(filepath).replace('.fits', '')
        fig.savefig(os.path.join(save_figures, f'{stem}_data_model_residual.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_figures, stem + '_data_model_residual.png')}")
    if doplot:
        plt.show()
    plt.close(fig)
    
    print(
        "Analysis complete. Optimal center: "
        f"({smart_fmt(xcen_opt)}, {smart_fmt(ycen_opt)})"
    )

    print(f"Flux in ring : {smart_fmt(norm)}")
    print(f"RMS residual: {smart_fmt(rms_residual)}")
    peak_leak = harmonic_fit['peak_value']
    print(f"Peak flare: {smart_fmt(peak_leak)}")
    peak_angle = harmonic_fit['peak_angle_deg']
    print(f"Peak angle: {smart_fmt(peak_angle)}")

    # Create MEF with analysis results
    # Copy header from input file and add new analysis parameters
    primary_hdu = fits.PrimaryHDU()
    
    # Copy all keywords from input file header (except structural keywords)
    # Suppress FITS verification warnings for non-standard keyword names
    structural_keywords = {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'PCOUNT', 'GCOUNT', 'XTENSION', 'EXTNAME', 'EXTVER', 'EXTLEVEL'}
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=VerifyWarning)
        for key in hdr:
            if key not in structural_keywords:
                try:
                    primary_hdu.header[key] = hdr[key]
                except:
                    pass  # Skip keywords that can't be copied
    
    # Add analysis results
    primary_hdu.header['FLUXRING'] = (norm,'Total flux in the ring model')
    primary_hdu.header['RMSRESI'] = (rms_residual,'RMS of residuals before normalization')
    primary_hdu.header['PEAKFLAR'] = (peak_leak,'Peak flare from harmonic fit, fraction of total flux')
    primary_hdu.header['ANGLFLAR'] = (peak_angle,'Angle of harmonic-fit peak flare')
    primary_hdu.header['XCEN'] = (xcen_opt,'Optimal X center')
    primary_hdu.header['YCEN'] = (ycen_opt,'Optimal Y center')
    
    # Extract RA/DEC from input file header
    try:
        ra = hdr.get('RA', 0.0)
        dec = hdr.get('DEC', 0.0)
    except:
        ra, dec = 0.0, 0.0
        print("Warning: Could not extract RA/DEC from input header, using defaults.")
    
    # Create WCS for the images
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [xcen_opt + 1, ycen_opt + 1]  # FITS uses 1-based indexing
    wcs.wcs.crval = [ra, dec]
    cdelt = CONFIG['wcs_cdelt']
    wcs.wcs.cdelt = [-cdelt, cdelt]  # Scale: negative RA for standard orientation
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    rotation_angle = CONFIG['wcs_rotation_angle']  # degrees
    cos_rot = np.cos(np.radians(rotation_angle))
    sin_rot = np.sin(np.radians(rotation_angle))
    wcs.wcs.cd = [[-cdelt * cos_rot,  cdelt * sin_rot],
                  [-cdelt * sin_rot, -cdelt * cos_rot]]

    diff = guiding_image - image2
    diff =  remove_hatch(diff)


    # Subtract column-wise robust mean efficiently
    for col in range(diff.shape[1]):
        col_data = diff[:, col]
        if np.any(~np.isnan(col_data)):
            diff[:, col] -= robust_mean(col_data)
    
    # Create image HDUs for guiding image and residual with WCS
    hdu1 = fits.ImageHDU(guiding_image0, name='GUIDING')
    hdu1.header.update(wcs.to_header())
    
    hdu2 = fits.ImageHDU(diff, name='RESIDUAL')
    hdu2.header.update(wcs.to_header())
    
    # Create table HDU for radial profile
    col1 = fits.Column(name='RADIUS', format='E', array=rad_profile)
    col2 = fits.Column(name='FLUX', format='E', array=profile)
    radial_table = fits.BinTableHDU.from_columns([col1, col2], name='RADIAL_PROFILE')
    
    # Create table HDU for angular profile
    col3 = fits.Column(name='ANGLE', format='E', array=theta_bin_center)
    col4 = fits.Column(name='RESIDUAL_FLUX', format='E', array=residual_flux_bin)
    angular_table = fits.BinTableHDU.from_columns([col3, col4], name='ANGULAR_PROFILE')
    
    # Write MEF with warnings suppressed
    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, radial_table, angular_table])
    
    # Create output folder if needed
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=VerifyWarning)
        hdul.writeto(output_path, overwrite=True)
    
    print(f"Output written to: {output_path}")
    return 'processed'




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze NIRPS guiding frames and extract PSF / RMS properties.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # single file
  python get_props_guiding.py NIRPS_2025-07-28T23_07_40_654.fits

  # explicit list, show plots
  python get_props_guiding.py --doplot file1.fits file2.fits

  # all FITS files under /data/raw/ written to /data/products/
  python get_props_guiding.py --base /data/raw/ --output /data/products/ "*.fits"

  # use base and output from guiding_config.yaml (no CLI overrides needed)
  python get_props_guiding.py "*.fits"

  # regenerate README documentation figures
  python get_props_guiding.py --documentation "*.fits"
        '''
    )

    parser.add_argument(
        'files', nargs='*',
        help='FITS file(s) or glob patterns to process. '
             'Patterns are expanded relative to --base (or data_folder in config).')
    parser.add_argument(
        '--base', type=str, default=None,
        help='Base directory prepended to every file / pattern argument. '
             'Falls back to user.<current_user>.data_folder in guiding_config.yaml '
             '(or user.default.data_folder).')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Folder where *_guiding_analysis.fits products are written. '
             'Falls back to user.<current_user>.output_folder in guiding_config.yaml '
             '(or user.default.output_folder); '
             'if that is also empty, results land next to the input files.')
    parser.add_argument('--doplot', action='store_true', default=False,
                        help='Display diagnostic plots interactively.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Reprocess even if the output file already exists.')
    parser.add_argument('--config', type=str, default=_CONFIG_PATH,
                        help=f'Path to YAML config file (default: {_CONFIG_PATH})')
    parser.add_argument('--documentation', action='store_true', default=False,
                        help='Save all diagnostic plots as PNG files to figures/ '
                             '(for README documentation). Implies --force.')

    args = parser.parse_args()

    # Reload config if a custom path was provided
    if args.config != _CONFIG_PATH:
        CONFIG = load_config(args.config)

    # ---- Resolve base directory (CLI > config > cwd) ----------------------
    base = args.base or CONFIG.get('data_folder') or ''
    base = os.path.expanduser(base)

    # ---- Resolve output directory (CLI > config > None = next to input) ---
    output_folder = args.output or CONFIG.get('output_folder') or None
    if output_folder:
        output_folder = os.path.expanduser(output_folder)

    # ---- Expand file patterns relative to base ----------------------------
    # Fall back to the wildcard from config when no files are given on the CLI.
    patterns = args.files
    if not patterns:
        config_wildcard = CONFIG.get('wildcard') or ''
        if config_wildcard:
            print(f"No files specified; using wildcard from config: '{config_wildcard}'")
            patterns = [config_wildcard]
        else:
            parser.error('Provide at least one FITS file or glob pattern '
                         '(or set wildcard in guiding_config.yaml).')

    filepaths = []
    for pattern in patterns:
        if base:
            full_pattern = os.path.join(base, pattern)
        else:
            full_pattern = pattern
        matched = sorted(glob.glob(full_pattern))
        if not matched:
            print(f"Warning: no files matched pattern '{full_pattern}'")
        filepaths.extend(matched)

    # Remove duplicates while preserving deterministic ordering.
    filepaths = sorted(set(filepaths))

    if not filepaths:
        print('No input files found. Exiting.')
        raise SystemExit(1)

    # ---- Documentation mode -----------------------------------------------
    _figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    save_figures = _figures_dir if args.documentation else None
    save_figures_basename = 'documentation' if args.documentation else None
    force = args.force or args.documentation

    # ---- Pre-filter by existing outputs/tokens when not forcing ----------
    n_found = len(filepaths)
    already_processed = 0
    pending_filepaths = []

    if not force:
        for filepath in filepaths:
            output_exists = os.path.exists(get_output_path(filepath, output_folder=output_folder))
            token_exists = os.path.exists(get_skip_token_path(filepath, output_folder=output_folder))
            if output_exists or token_exists:
                already_processed += 1
            else:
                pending_filepaths.append(filepath)
    else:
        pending_filepaths = filepaths

    # ---- Pre-run summary --------------------------------------------------
    n_total = len(pending_filepaths)
    print('=' * 60)
    print('  NIRPS Guiding Analysis')
    print('=' * 60)
    print(f'  Files found            : {n_found}')
    print(f'  Already processed      : {already_processed}')
    print(f'  Files to process now   : {n_total}')
    print(f'  Base directory   : {base or "(current directory)"}')
    print(f'  Output directory : {output_folder or "(next to each input file)"}')
    print(f'  Force reprocess  : {force}')
    print(f'  Show plots       : {args.doplot}')
    print('=' * 60)

    if n_total == 0:
        print('No pending files to process. Exiting.')
        raise SystemExit(0)

    # ---- Process each file ------------------------------------------------
    # Output from each file is captured and shown while the *next* file runs,
    # so the info stays on screen until we naturally reach the next iteration.
    n_done = 0
    elapsed_times = []
    pending_display = []  # lines from the previous file, shown at next iteration

    for filepath in pending_filepaths:
        n_done += 1
        n_remaining = n_total - n_done

        # ETA based on average elapsed time so far
        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            eta_sec = avg * (n_remaining + 1)
            eta_str = format_duration(eta_sec)
            timing_line = f'  Avg per file: {smart_fmt(avg)}s  |  Est. time left: {eta_str}'
        else:
            timing_line = ''

        # Clear screen, replay previous file output, ETA, closing separator
        print('\033[2J\033[H', end='')
        for line in pending_display:
            print(line)
        if timing_line:
            print(timing_line)
        if pending_display:
            print('=' * 60)

        file_header = [
            '=' * 60,
            f'  File {n_done}/{n_total}  |  done: {n_done - 1}  |  to do: {n_remaining}',
            f'  Total files: {n_found}  |  already processed: {already_processed}',
            f'  {os.path.basename(filepath)}',
            '=' * 60,
        ]

        if not os.path.exists(filepath):
            pending_display = file_header + [f'Error: File not found: {filepath}']
            continue

        # Capture all output from analyze_guiding_image into a buffer
        buf = io.StringIO()
        start_time = time.time()
        with redirect_stdout(buf):
            status = analyze_guiding_image(
                filepath,
                doplot=args.doplot,
                output_folder=output_folder,
                force=force,
                save_figures=save_figures,
                save_figures_basename=save_figures_basename,
            )
        elapsed_time = time.time() - start_time
        if status == 'processed':
            elapsed_times.append(elapsed_time)

        captured = buf.getvalue().rstrip('\n').split('\n') if buf.getvalue() else []
        status_line = f'Status: {status}' if status else 'Status: unknown'
        pending_display = file_header + captured + [status_line, f'Completed in {smart_fmt(elapsed_time)} seconds']

    # Show the last file's output after the loop
    print('\033[2J\033[H', end='')
    for line in pending_display:
        print(line)
    print('=' * 60)
    print(f'  Finished {n_total} pending file(s).')
    print(f'  Newly processed outputs: {len(elapsed_times)}')
    print('=' * 60)


