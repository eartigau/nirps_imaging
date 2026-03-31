"""
Merge guiding images by object name.

This script reads guiding_config.yaml, finds FITS files for configured objects by
parsing headers, aligns images to the median centroid position using integer
pixel shifts, and writes median merged FOV products.

Input files are discovered only in the configured output_folder and must match
*_guiding_analysis.fits.

Usage:
    python merge_guiding_objects.py
    python merge_guiding_objects.py --config guiding_config.yaml
    python merge_guiding_objects.py --objects Proxima "Barnard's star"
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from astropy.io import fits


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guiding_config.yaml")
_DEFAULTS = {
    "merge_objects": [],
}


def normalize_object_name(name: str) -> str:
    """Normalize object names for robust matching."""

    return " ".join(str(name).strip().lower().split())


def load_config(config_path: str = _CONFIG_PATH) -> dict:
    """Load YAML config and resolve current user paths."""

    cfg = dict(_DEFAULTS)
    cfg["data_folder"] = ""
    cfg["output_folder"] = ""
    cfg["wildcard"] = ""

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    for key, value in loaded.items():
        if key != "user":
            cfg[key] = value

    users_cfg = loaded.get("user", {})
    if not isinstance(users_cfg, dict):
        raise ValueError("'user' section in config must be a mapping")

    default_user_cfg = users_cfg.get("default", {})
    if default_user_cfg is None:
        default_user_cfg = {}
    if not isinstance(default_user_cfg, dict):
        raise ValueError("'user.default' section in config must be a mapping")

    current_user = getpass.getuser()
    selected_user_cfg = users_cfg.get(current_user)

    if selected_user_cfg is None:
        for key, entry in users_cfg.items():
            if key == "default":
                continue
            if fnmatch.fnmatch(current_user, key):
                selected_user_cfg = entry
                break

    if selected_user_cfg is None:
        selected_user_cfg = {}

    if not isinstance(selected_user_cfg, dict):
        raise ValueError(f"Matched user entry for '{current_user}' must be a mapping")

    merged_user_cfg = dict(default_user_cfg)
    merged_user_cfg.update(selected_user_cfg)

    cfg["data_folder"] = merged_user_cfg.get("data_folder", "")
    cfg["output_folder"] = merged_user_cfg.get("output_folder", "")
    cfg["wildcard"] = merged_user_cfg.get("wildcard", "")

    merge_objects = cfg.get("merge_objects", [])
    if merge_objects is None:
        merge_objects = []
    if not isinstance(merge_objects, list):
        raise ValueError("'merge_objects' in config must be a list")

    cfg["merge_objects"] = [str(obj).strip() for obj in merge_objects if str(obj).strip()]
    return cfg


def get_fits_candidates(base: str, wildcard: str) -> List[str]:
    """Find candidate analysis FITS files inside output_folder."""

    if not base:
        return []

    pattern = wildcard or "*_guiding_analysis.fits"
    full_pattern = os.path.join(base, pattern)
    return sorted(set(glob.glob(full_pattern)))


def extract_object_from_header(header: fits.Header) -> Optional[str]:
    """Return best-effort object name from standard header keys."""

    candidate_keys = ["OBJECT", "OBJNAME", "TARGNAME", "TARGET", "ESO OBS TARG NAME"]
    for key in candidate_keys:
        value = header.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def read_guiding_image(filepath: str) -> Tuple[np.ndarray, fits.Header, str]:
    """
    Read a 2D guiding image and primary header from FITS file.

    Returns
    -------
    image : ndarray
        Guiding image data as float array.
    header : fits.Header
        Primary header used for object selection and metadata.
    image_ext : str
        Extension label used for provenance.
    """

    with fits.open(filepath, memmap=False) as hdul:
        header = hdul[0].header.copy()

        # Preferred: explicit GUIDING extension by name.
        for hdu in hdul:
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            if extname == "GUIDING" and getattr(hdu, "data", None) is not None:
                data = np.asarray(hdu.data, dtype=float)
                if data.ndim == 2:
                    return data, header, "GUIDING"

        # Fallback 1: primary HDU if 2D.
        if getattr(hdul[0], "data", None) is not None:
            data0 = np.asarray(hdul[0].data, dtype=float)
            if data0.ndim == 2:
                return data0, header, "PRIMARY"

        # Fallback 2: first 2D image extension.
        for idx, hdu in enumerate(hdul[1:], start=1):
            if getattr(hdu, "data", None) is None:
                continue
            datai = np.asarray(hdu.data, dtype=float)
            if datai.ndim == 2:
                return datai, header, f"HDU{idx}"

    raise ValueError(f"No 2D image found in {filepath}")


def robust_centroid(image: np.ndarray) -> Tuple[float, float]:
    """Estimate centroid from positive flux above a robust background."""

    work = np.asarray(image, dtype=float).copy()
    finite = np.isfinite(work)
    if not np.any(finite):
        raise ValueError("Image has no finite pixels")

    background = np.nanmedian(work)
    work -= background
    work[~np.isfinite(work)] = np.nan

    # Keep only positive residual flux to avoid centroid bias from background noise.
    work[work < 0] = 0.0

    flux_sum = np.nansum(work)
    if not np.isfinite(flux_sum) or flux_sum <= 0:
        y_mid = (image.shape[0] - 1) / 2.0
        x_mid = (image.shape[1] - 1) / 2.0
        return x_mid, y_mid

    y_idx, x_idx = np.indices(image.shape, dtype=float)
    xcen = np.nansum(work * x_idx) / flux_sum
    ycen = np.nansum(work * y_idx) / flux_sum
    return float(xcen), float(ycen)


def roll_with_nan_padding(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift image with np.roll, then blank wrapped edges with NaN."""

    rolled = np.roll(image, shift=(dy, dx), axis=(0, 1)).astype(float, copy=False)

    if dy > 0:
        rolled[:dy, :] = np.nan
    elif dy < 0:
        rolled[dy:, :] = np.nan

    if dx > 0:
        rolled[:, :dx] = np.nan
    elif dx < 0:
        rolled[:, dx:] = np.nan

    return rolled


def make_merged_product(
    object_name: str,
    selected_files: List[str],
    output_folder_merged: str,
) -> Optional[str]:
    """Build registered cube and median FOV for one object."""

    if not selected_files:
        return None

    images: List[np.ndarray] = []
    centers: List[Tuple[float, float]] = []
    used_files: List[str] = []

    shape_ref: Optional[Tuple[int, int]] = None

    for filepath in selected_files:
        try:
            image, _header, _ext = read_guiding_image(filepath)
        except Exception as exc:
            print(f"  Skip unreadable file: {filepath} ({exc})")
            continue

        if shape_ref is None:
            shape_ref = image.shape
        if image.shape != shape_ref:
            print(
                f"  Skip shape mismatch: {filepath} has {image.shape}, expected {shape_ref}"
            )
            continue

        try:
            xcen, ycen = robust_centroid(image)
        except Exception as exc:
            print(f"  Skip centroid failure: {filepath} ({exc})")
            continue

        images.append(np.asarray(image, dtype=float))
        centers.append((xcen, ycen))
        used_files.append(filepath)

    if not images:
        return None

    centers_arr = np.asarray(centers, dtype=float)
    x_med = float(np.nanmedian(centers_arr[:, 0]))
    y_med = float(np.nanmedian(centers_arr[:, 1]))

    registered_cube = np.full((len(images), shape_ref[0], shape_ref[1]), np.nan, dtype=float)

    shifts: List[Tuple[int, int]] = []
    for idx, (image, (xcen, ycen)) in enumerate(zip(images, centers)):
        dx = int(np.rint(x_med - xcen))
        dy = int(np.rint(y_med - ycen))
        shifts.append((dy, dx))
        registered_cube[idx] = roll_with_nan_padding(image, dy=dy, dx=dx)

    median_fov = np.nanmedian(registered_cube, axis=0)

    os.makedirs(output_folder_merged, exist_ok=True)
    object_tag = "_".join(object_name.strip().split())
    output_filename = f"{object_tag}_guiding_median_fov.fits"
    output_path = os.path.join(output_folder_merged, output_filename)

    hdr = fits.Header()
    hdr["OBJECT"] = object_name
    hdr["NINPUT"] = len(used_files)
    hdr["XCENMED"] = (x_med, "Median X centroid before registration")
    hdr["YCENMED"] = (y_med, "Median Y centroid before registration")
    hdr["COMMENT"] = "Images aligned with np.roll and NaN-padded wrapped edges"

    primary_hdu = fits.PrimaryHDU(header=hdr)
    median_hdu = fits.ImageHDU(median_fov, name="MEDIAN_FOV")
    cube_hdu = fits.ImageHDU(registered_cube, name="REGISTERED_CUBE")

    col_file = fits.Column(name="FILE", format="A256", array=np.asarray(used_files, dtype="S256"))
    col_x = fits.Column(name="XCEN", format="D", array=np.asarray([c[0] for c in centers]))
    col_y = fits.Column(name="YCEN", format="D", array=np.asarray([c[1] for c in centers]))
    col_dy = fits.Column(name="SHIFT_DY", format="J", array=np.asarray([s[0] for s in shifts], dtype=np.int32))
    col_dx = fits.Column(name="SHIFT_DX", format="J", array=np.asarray([s[1] for s in shifts], dtype=np.int32))
    provenance_hdu = fits.BinTableHDU.from_columns(
        [col_file, col_x, col_y, col_dy, col_dx],
        name="PROVENANCE",
    )

    fits.HDUList([primary_hdu, median_hdu, cube_hdu, provenance_hdu]).writeto(
        output_path,
        overwrite=True,
    )

    return output_path


def group_files_by_object(filepaths: List[str]) -> Dict[str, List[str]]:
    """Group files by normalized object name from primary headers."""

    grouped: Dict[str, List[str]] = {}

    for filepath in filepaths:
        try:
            header = fits.getheader(filepath, 0)
        except Exception as exc:
            print(f"Skip unreadable header: {filepath} ({exc})")
            continue

        object_name = extract_object_from_header(header)
        if object_name is None:
            print(f"Skip missing object keyword: {filepath}")
            continue

        key = normalize_object_name(object_name)
        grouped.setdefault(key, []).append(filepath)

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge guiding images by target object using YAML-configured object list."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_CONFIG_PATH,
        help=f"Path to YAML config (default: {_CONFIG_PATH})",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Optional object list override (default: merge_objects from YAML)",
    )
    parser.add_argument(
        "--scan-pattern",
        type=str,
        default="*_guiding_analysis.fits",
        help="Glob pattern used inside output folder (default: *_guiding_analysis.fits)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output folder override before adding '_merged' suffix "
            "(default: user-resolved output_folder in YAML). "
            "This folder is also where input *_guiding_analysis.fits are scanned."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    objects = args.objects if args.objects is not None else cfg.get("merge_objects", [])
    objects = [str(obj).strip() for obj in objects if str(obj).strip()]
    if not objects:
        raise SystemExit(
            "No objects to merge. Add entries under 'merge_objects' in YAML "
            "or pass --objects on the command line."
        )

    out_base = args.output if args.output is not None else (cfg.get("output_folder") or "")
    if not out_base:
        raise SystemExit(
            "No output_folder configured. Set user.<name>.output_folder in YAML "
            "or pass --output."
        )

    out_base = os.path.expanduser(out_base)
    output_folder_merged = f"{out_base}_merged"

    filepaths = get_fits_candidates(base=out_base, wildcard=args.scan_pattern)
    if not filepaths:
        raise SystemExit(
            f"No candidate FITS files found in output folder '{out_base}' "
            f"with pattern '{args.scan_pattern}'."
        )

    grouped = group_files_by_object(filepaths)

    print("=" * 60)
    print("Guiding Object Merge")
    print("=" * 60)
    print(f"Candidate files scanned: {len(filepaths)}")
    print(f"Input folder           : {out_base}")
    print(f"Output merged folder   : {output_folder_merged}")
    print(f"Objects requested      : {', '.join(objects)}")
    print("=" * 60)

    merged_count = 0
    for obj in objects:
        key = normalize_object_name(obj)
        selected = grouped.get(key, [])
        print(f"Object '{obj}': {len(selected)} matching files")

        if not selected:
            continue

        output_path = make_merged_product(
            object_name=obj,
            selected_files=selected,
            output_folder_merged=output_folder_merged,
        )

        if output_path is None:
            print("  No valid images available after filtering.")
            continue

        merged_count += 1
        print(f"  Wrote merged product: {output_path}")

    print("=" * 60)
    print(f"Merged products written: {merged_count}/{len(objects)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
