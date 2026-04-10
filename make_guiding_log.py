"""
NIRPS Guiding Analysis — CSV Log Generator
===========================================

Context
-------
This script scans the processed NIRPS guiding-analysis FITS products
(`*_guiding_analysis.fits`) and builds a single CSV log of the main header
measurements written by `get_props_guiding.py`.

It can also attach throughput information derived from the corresponding
extracted APERO spectra (`*_pp_e2dsff_A.fits`). The extracted path is
predicted from the guiding filename using the configured extracted root and
the NIRPS night-folder convention. Existing throughput cache rows are loaded
first; if the extracted spectrum exists locally and no cache row is found,
the throughput metrics are computed on the fly using the same method as the
`nirps_throughput` project.

Repository
----------
https://github.com/eartigau/nirps_imaging
"""

import csv
from datetime import datetime, timedelta
import fnmatch
import getpass
import glob
import os

from astropy.io import fits
import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'guiding_config.yaml')

BASE_COLUMNS = [
    'file',
    'object',
    'mjd',
    'date',
    'fluxring',
    'rmsresi',
    'peakflar',
    'anglflar',
    'xcen',
    'ycen',
]

THROUGHPUT_CACHE_COLUMNS = [
    'filename',
    'date_obs',
    'mjd_obs',
    'object',
    'jmag',
    'exptime',
    'airmass',
    'median_flux_e',
    'flux_per_sec',
    'throughput_proxy',
    'n_orders_used',
    'zero_point',
]

DEFAULT_THROUGHPUT = {
    'enabled': False,
    'guiding_suffix': '_guiding_analysis.fits',
    'raw_suffix': '.fits',
    'extracted_suffix': '_pp_e2dsff_A.fits',
    'night_folder_offset_hours': 12,
    'band_min_nm': 1170.0,
    'band_max_nm': 1330.0,
    'band_label': 'J',
    'blaze_percentile': 50,
    'jmag_keyword': 'HIERARCH ESO OCS TARG JMAG',
    'date_obs_keyword': 'DATE-OBS',
    'mjd_obs_keyword': 'MJD-OBS',
    'exptime_keyword': 'EXPTIME',
    'object_keyword': 'DRSOBJN',
    'airmass_keyword': 'HIERARCH ESO TEL AIRM START',
    'spec_extension': 'EXT_E2DS_FF',
    'blaze_extension': 'FF_BLAZE',
    'wave_extension': 'WAVE_NIGHT',
    'merge_columns': [
        'date_obs',
        'mjd_obs',
        'jmag',
        'exptime',
        'airmass',
        'median_flux_e',
        'flux_per_sec',
        'throughput_proxy',
        'n_orders_used',
        'zero_point',
    ],
    'extracted_root': '',
    'wave_file': '',
    'blaze_file': '',
    'results_csv': '',
    'bootstrap_results_csv': '',
}


def resolve_user_mapping(users_cfg):
    """Resolve a user-specific mapping using exact and wildcard matches."""

    if not isinstance(users_cfg, dict):
        return {}

    default_cfg = users_cfg.get('default', {}) or {}
    if not isinstance(default_cfg, dict):
        default_cfg = {}

    current_user = getpass.getuser()
    selected_cfg = users_cfg.get(current_user)

    if selected_cfg is None:
        for key, value in users_cfg.items():
            if key == 'default':
                continue
            if fnmatch.fnmatch(current_user, key):
                selected_cfg = value
                break

    if selected_cfg is None or not isinstance(selected_cfg, dict):
        selected_cfg = {}

    merged = dict(default_cfg)
    merged.update(selected_cfg)
    return merged


def load_config(config_path=CONFIG_PATH):
    """Load guiding and throughput configuration from YAML."""

    cfg = {
        'data_folder': '',
        'output_folder': '',
        'wildcard': '',
        'throughput': dict(DEFAULT_THROUGHPUT),
    }

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}.")
        return cfg

    with open(config_path, 'r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle) or {}

    user_cfg = resolve_user_mapping(loaded.get('user', {}))
    cfg['data_folder'] = user_cfg.get('data_folder', '')
    cfg['output_folder'] = user_cfg.get('output_folder', '')
    cfg['wildcard'] = user_cfg.get('wildcard', '')

    throughput_cfg = dict(DEFAULT_THROUGHPUT)
    loaded_throughput = loaded.get('throughput', {}) or {}
    if isinstance(loaded_throughput, dict):
        for key, value in loaded_throughput.items():
            if key != 'user':
                throughput_cfg[key] = value
        throughput_user_cfg = resolve_user_mapping(loaded_throughput.get('user', {}))
        throughput_cfg.update(throughput_user_cfg)

    if not throughput_cfg.get('results_csv') and cfg['output_folder']:
        throughput_cfg['results_csv'] = os.path.join(cfg['output_folder'], 'throughput_results.csv')

    cfg['throughput'] = throughput_cfg
    return cfg


def iter_with_progress(items, description):
    """Yield items with a progress bar when tqdm is available."""

    if tqdm is not None:
        yield from tqdm(items, desc=description, unit='file')
        return

    total = len(items)
    for index, item in enumerate(items, start=1):
        print(f"{description}: {index}/{total}", end='\r', flush=True)
        yield item
    print(' ' * 80, end='\r')


def row_sort_key(row):
    """Sort by guiding DATE, then MJD, then filename."""

    date_value = str(row.get('date', '') or '')
    mjd_value = row.get('mjd', '')
    try:
        mjd_value = float(mjd_value) if mjd_value != '' else 0.0
    except (TypeError, ValueError):
        mjd_value = 0.0
    return (date_value, mjd_value, row.get('file', ''))


def throughput_cache_sort_key(row):
    """Sort throughput cache rows by observation date, then MJD, then filename."""

    date_value = str(row.get('date_obs', '') or '')
    mjd_value = row.get('mjd_obs', '')
    try:
        mjd_value = float(mjd_value) if mjd_value != '' else 0.0
    except (TypeError, ValueError):
        mjd_value = 0.0
    return (date_value, mjd_value, row.get('filename', ''))


def get_guiding_output_files(output_folder):
    """Return all guiding-analysis FITS products in the configured output folder."""

    pattern = os.path.join(output_folder, '*_guiding_analysis.fits')
    return sorted(glob.glob(pattern))


def build_extracted_basename(guiding_filename, throughput_cfg):
    """Map a guiding-analysis filename to its extracted-spectrum basename."""

    guiding_suffix = throughput_cfg['guiding_suffix']
    raw_suffix = throughput_cfg['raw_suffix']
    extracted_suffix = throughput_cfg['extracted_suffix']

    if not guiding_filename.endswith(guiding_suffix):
        return None, None

    raw_basename = guiding_filename[:-len(guiding_suffix)] + raw_suffix
    if not raw_basename.endswith(raw_suffix):
        return raw_basename, None

    extracted_basename = raw_basename[:-len(raw_suffix)] + extracted_suffix
    return raw_basename, extracted_basename


def infer_night_folder(raw_basename, offset_hours):
    """Infer the NIRPS night folder from the raw filename timestamp."""

    stamp = os.path.splitext(raw_basename)[0]
    if not stamp.startswith('NIRPS_'):
        return None

    stamp = stamp[len('NIRPS_'):]
    try:
        observation_time = datetime.strptime(stamp, '%Y-%m-%dT%H_%M_%S_%f')
    except ValueError:
        return None

    night_time = observation_time - timedelta(hours=float(offset_hours))
    return night_time.strftime('%Y-%m-%d')


def predict_extracted_path(guiding_filename, throughput_cfg):
    """Predict the extracted spectrum path associated with a guiding product."""

    raw_basename, extracted_basename = build_extracted_basename(guiding_filename, throughput_cfg)
    if raw_basename is None or extracted_basename is None:
        return None, None

    night_folder = infer_night_folder(raw_basename, throughput_cfg['night_folder_offset_hours'])
    if night_folder is None:
        return None, extracted_basename

    extracted_root = throughput_cfg.get('extracted_root', '')
    if not extracted_root:
        return None, extracted_basename

    return os.path.join(extracted_root, night_folder, extracted_basename), extracted_basename


def load_csv_cache(csv_path, key_column):
    """Load a CSV cache as a dict keyed by one of its columns."""

    if not csv_path or not os.path.exists(csv_path):
        return {}

    with open(csv_path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or key_column not in reader.fieldnames:
            return {}
        return {row[key_column]: row for row in reader if row.get(key_column)}


def write_throughput_cache(csv_path, cache_rows):
    """Write the local throughput cache used by this repository."""

    if not csv_path:
        return

    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    rows = sorted(cache_rows.values(), key=throughput_cache_sort_key)
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=THROUGHPUT_CACHE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_wave_grid(throughput_cfg):
    """Return the wavelength grid [n_orders x n_pixels] in nm."""

    with fits.open(throughput_cfg['wave_file']) as hdul:
        return hdul[throughput_cfg['wave_extension']].data.astype(float)


def load_blaze(throughput_cfg):
    """Return the blaze function [n_orders x n_pixels]."""

    with fits.open(throughput_cfg['blaze_file']) as hdul:
        return hdul[throughput_cfg['blaze_extension']].data.astype(float)


def find_band_orders(wave, band_min, band_max):
    """Return the order indices whose median wavelength falls inside the band."""

    orders = []
    for index in range(wave.shape[0]):
        median_wave = np.nanmedian(wave[index])
        if band_min <= median_wave <= band_max:
            orders.append(index)
    return orders


def build_blaze_masks(blaze, orders, percentile):
    """Build a per-order blaze mask based on a percentile threshold."""

    masks = {}
    for order in orders:
        blaze_profile = blaze[order]
        threshold = np.nanpercentile(blaze_profile, percentile)
        masks[order] = blaze_profile >= threshold
    return masks


def process_extracted_spectrum(filepath, throughput_cfg, orders, blaze_masks):
    """Compute throughput metrics for one extracted spectrum."""

    try:
        with fits.open(filepath) as hdul:
            header = hdul[0].header
            flux = hdul[throughput_cfg['spec_extension']].data.astype(float)
    except Exception as exc:
        print(f"Skipping throughput for {os.path.basename(filepath)}: {exc}")
        return None

    jmag = header.get(throughput_cfg['jmag_keyword'])
    exptime = header.get(throughput_cfg['exptime_keyword'], np.nan)
    if jmag is None or exptime is None:
        return None

    try:
        exptime = float(exptime)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(exptime) or exptime <= 0:
        return None

    order_medians = []
    for order in orders:
        mask = blaze_masks[order]
        good_flux = flux[order][mask]
        if good_flux.size == 0:
            continue
        median_flux = np.nanmedian(good_flux)
        if np.isfinite(median_flux) and median_flux > 0:
            order_medians.append(median_flux)

    if not order_medians:
        return None

    median_flux_e = float(np.mean(order_medians))
    flux_per_sec = median_flux_e / exptime
    throughput_proxy = flux_per_sec * 10 ** (float(jmag) / 2.5)
    zero_point = 2.5 * np.log10(throughput_proxy) if throughput_proxy > 0 else np.nan
    airmass = header.get(throughput_cfg['airmass_keyword'], np.nan)
    try:
        airmass = float(airmass)
    except (TypeError, ValueError):
        airmass = np.nan
    obj = header.get(throughput_cfg['object_keyword']) or header.get('OBJECT', '')

    return {
        'filename': os.path.basename(filepath),
        'date_obs': header.get(throughput_cfg['date_obs_keyword'], ''),
        'mjd_obs': header.get(throughput_cfg['mjd_obs_keyword'], np.nan),
        'object': obj,
        'jmag': float(jmag),
        'exptime': exptime,
        'airmass': airmass,
        'median_flux_e': median_flux_e,
        'flux_per_sec': flux_per_sec,
        'throughput_proxy': throughput_proxy,
        'n_orders_used': len(order_medians),
        'zero_point': zero_point,
    }


def compute_missing_throughput(missing_paths, throughput_cfg):
    """Compute throughput rows for extracted spectra that exist but are not cached."""

    if not missing_paths:
        return {}

    required_files = [throughput_cfg.get('wave_file', ''), throughput_cfg.get('blaze_file', '')]
    if not all(path and os.path.exists(path) for path in required_files):
        print('Warning: throughput reference files are missing; skipping on-the-fly throughput computation.')
        return {}

    wave = load_wave_grid(throughput_cfg)
    blaze = load_blaze(throughput_cfg)
    orders = find_band_orders(wave, throughput_cfg['band_min_nm'], throughput_cfg['band_max_nm'])
    if not orders:
        print('Warning: no throughput orders overlap the configured wavelength band.')
        return {}

    blaze_masks = build_blaze_masks(blaze, orders, throughput_cfg['blaze_percentile'])
    computed = {}
    for filepath in iter_with_progress(missing_paths, 'Computing throughput'):
        result = process_extracted_spectrum(filepath, throughput_cfg, orders, blaze_masks)
        if result is not None:
            computed[result['filename']] = result
    return computed


def build_throughput_cache(extracted_candidates, throughput_cfg):
    """Build the merged throughput cache from bootstrap, local cache, and new computations."""

    if not throughput_cfg.get('enabled', False):
        return {}

    bootstrap_cache = load_csv_cache(throughput_cfg.get('bootstrap_results_csv', ''), 'filename')
    local_cache = load_csv_cache(throughput_cfg.get('results_csv', ''), 'filename')

    merged_cache = dict(bootstrap_cache)
    merged_cache.update(local_cache)

    missing_paths = []
    for extracted_filename, extracted_path in extracted_candidates.items():
        if extracted_filename in merged_cache:
            continue
        if extracted_path and os.path.exists(extracted_path):
            missing_paths.append(extracted_path)

    computed_cache = compute_missing_throughput(missing_paths, throughput_cfg)
    if computed_cache:
        local_cache.update(computed_cache)
        merged_cache.update(computed_cache)

    if throughput_cfg.get('results_csv', ''):
        for extracted_filename in extracted_candidates:
            if extracted_filename in merged_cache and extracted_filename not in local_cache:
                local_cache[extracted_filename] = merged_cache[extracted_filename]
        if local_cache:
            write_throughput_cache(throughput_cfg.get('results_csv', ''), local_cache)

    return merged_cache


def main():
    """Create the guiding log and merge any available throughput information."""

    config = load_config()
    output_folder = config['output_folder']
    if not output_folder or not os.path.isdir(output_folder):
        print(f"Output folder not found or not configured: '{output_folder}'")
        return

    guiding_files = get_guiding_output_files(output_folder)
    if not guiding_files:
        print(f"No *_guiding_analysis.fits files found in {output_folder}")
        return

    rows = []
    extracted_candidates = {}
    throughput_cfg = config['throughput']

    for filepath in iter_with_progress(guiding_files, 'Reading FITS headers'):
        try:
            header = fits.getheader(filepath, 0)
        except Exception as exc:
            print(f"Skipping {os.path.basename(filepath)}: {exc}")
            continue

        row = {
            'file': os.path.basename(filepath),
            'object': header.get('OBJECT', ''),
            'mjd': header.get('MJD-OBS', ''),
            'date': header.get('DATE', ''),
            'fluxring': header.get('FLUXRING', ''),
            'rmsresi': header.get('RMSRESI', ''),
            'peakflar': header.get('PEAKFLAR', ''),
            'anglflar': header.get('ANGLFLAR', ''),
            'xcen': header.get('XCEN', ''),
            'ycen': header.get('YCEN', ''),
        }

        extracted_path, extracted_filename = predict_extracted_path(row['file'], throughput_cfg)
        row['_throughput_filename'] = extracted_filename or ''
        if extracted_filename:
            extracted_candidates.setdefault(extracted_filename, extracted_path)

        rows.append(row)

    throughput_cache = build_throughput_cache(extracted_candidates, throughput_cfg)
    throughput_columns = [column for column in throughput_cfg.get('merge_columns', []) if column not in BASE_COLUMNS]

    matched_throughput = 0
    for row in rows:
        throughput_row = throughput_cache.get(row.pop('_throughput_filename', ''))
        if throughput_row is not None:
            matched_throughput += 1
            if not row.get('object'):
                row['object'] = throughput_row.get('object', '')
            for column in throughput_columns:
                row[column] = throughput_row.get(column, '')
        else:
            for column in throughput_columns:
                row[column] = ''

    rows.sort(key=row_sort_key)

    csv_columns = BASE_COLUMNS + throughput_columns
    csv_path = os.path.join(output_folder, 'guiding_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged throughput metrics for {matched_throughput}/{len(rows)} guiding files")
    print(f"Wrote {len(rows)} entries to {csv_path}")


if __name__ == '__main__':
    main()
