"""
NIRPS Guiding Analysis — CSV Log Generator
===========================================

Creates a CSV log summarising every processed guiding-analysis FITS product
found in the configured output folder.  Each row corresponds to one
*_guiding_analysis.fits file and records the key header keywords written by
get_props_guiding.py:

    file       — base filename of the analysis product
    object     — target name                       (OBJECT)
    mjd        — Modified Julian Date              (MJD-OBS)
    date       — human-readable UTC date-time      (DATE)
    fluxring   — integrated flux inside the ring   (FLUXRING)
    rmsresi    — RMS of the radial residuals       (RMSRESI)
    peakflar   — peak amplitude of flaring         (PEAKFLAR)
    anglflar   — position angle of the flare [deg] (ANGLFLAR)
    xcen       — ring centre X position [px]       (XCEN)
    ycen       — ring centre Y position [px]       (YCEN)

The output CSV is written to the same output folder and is sorted by MJD.

Configuration (data/output folders) is read from guiding_config.yaml using
the same user-resolution logic as the main pipeline.

Repository: https://github.com/eartigau/nirps_imaging
"""

import csv
import glob
import os

from astropy.io import fits
import yaml
import getpass
import fnmatch

# ---------------------------------------------------------------------------
# Configuration loading (mirrors get_props_guiding.py logic)
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'guiding_config.yaml')


def load_folders(config_path=_CONFIG_PATH):
    """Return (output_folder,) resolved from the YAML user map."""
    output_folder = ''
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}.")
        return output_folder

    with open(config_path, 'r') as fh:
        loaded = yaml.safe_load(fh) or {}

    users_cfg = loaded.get('user', {})
    if not isinstance(users_cfg, dict):
        return output_folder

    default_user_cfg = users_cfg.get('default', {}) or {}
    current_user = getpass.getuser()
    selected_user_cfg = users_cfg.get(current_user)

    if selected_user_cfg is None:
        reserved = {'default'}
        for key in users_cfg:
            if key not in reserved and fnmatch.fnmatch(current_user, key):
                selected_user_cfg = users_cfg[key]
                break
        if selected_user_cfg is None:
            selected_user_cfg = {}

    if not isinstance(selected_user_cfg, dict):
        selected_user_cfg = {}

    merged = dict(default_user_cfg)
    merged.update(selected_user_cfg)
    return merged.get('output_folder', '')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_folder = load_folders()
    if not output_folder or not os.path.isdir(output_folder):
        print(f"Output folder not found or not configured: '{output_folder}'")
        return

    pattern = os.path.join(output_folder, '*_guiding_analysis.fits')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No *_guiding_analysis.fits files found in {output_folder}")
        return

    header_keys = ['OBJECT', 'MJD-OBS', 'DATE', 'FLUXRING', 'RMSRESI',
                   'PEAKFLAR', 'ANGLFLAR', 'XCEN', 'YCEN']
    csv_columns = ['file', 'object', 'mjd', 'date', 'fluxring', 'rmsresi',
                   'peakflar', 'anglflar', 'xcen', 'ycen']

    rows = []
    for fpath in files:
        try:
            hdr = fits.getheader(fpath, 0)
        except Exception as e:
            print(f"Skipping {os.path.basename(fpath)}: {e}")
            continue

        row = {'file': os.path.basename(fpath)}
        for key, col in zip(header_keys, csv_columns[1:]):
            row[col] = hdr.get(key, '')
        rows.append(row)

    # Sort by MJD
    rows.sort(key=lambda r: float(r['mjd']) if r['mjd'] != '' else 0.0)

    csv_path = os.path.join(output_folder, 'guiding_log.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {csv_path}")


if __name__ == '__main__':
    main()
