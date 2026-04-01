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


def read_analysis_images(filepath: str) -> Tuple[np.ndarray, np.ndarray, float, float, fits.Header]:
    """
    Read GUIDING and RESIDUAL images plus the optimizer-derived center
    (XCEN/YCEN keywords written by get_props_guiding.py) from a
    *_guiding_analysis.fits file.

    Returns
    -------
    guiding : ndarray
        GUIDING extension (raw image before radial-profile subtraction).
    residual : ndarray
        RESIDUAL extension (image after radial-profile subtraction).
    xcen : float
        Optimal X center (column) stored in primary header keyword XCEN.
    ycen : float
        Optimal Y center (row) stored in primary header keyword YCEN.
    header : fits.Header
        Primary header (used for object name and metadata).
    """

    with fits.open(filepath, memmap=False) as hdul:
        header = hdul[0].header.copy()

        guiding = None
        residual = None

        for hdu in hdul:
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            if extname == "GUIDING" and getattr(hdu, "data", None) is not None:
                data = np.asarray(hdu.data, dtype=float)
                if data.ndim == 2:
                    guiding = data
            if extname == "RESIDUAL" and getattr(hdu, "data", None) is not None:
                data = np.asarray(hdu.data, dtype=float)
                if data.ndim == 2:
                    residual = data

    if guiding is None:
        raise ValueError(f"Missing 2D GUIDING extension in {filepath}")
    if residual is None:
        raise ValueError(f"Missing 2D RESIDUAL extension in {filepath}")
    if guiding.shape != residual.shape:
        raise ValueError(
            f"GUIDING/RESIDUAL shape mismatch in {filepath}: "
            f"{guiding.shape} vs {residual.shape}"
        )

    xcen = header.get("XCEN")
    ycen = header.get("YCEN")
    if xcen is None or ycen is None:
        raise ValueError(
            f"Missing XCEN/YCEN keywords in primary header of {filepath}. "
            "Re-run get_props_guiding.py to regenerate the analysis products."
        )

    return guiding, residual, float(xcen), float(ycen), header


def integer_shift_with_nan_padding(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Shift a 2D image by integer pixel offsets using np.roll and replace the
    wrapped edge pixels that were rolled in with NaN.

    Parameters
    ----------
    image : ndarray
        2D input image (float).  May contain NaN.
    dy : int
        Shift along axis 0 (rows).  Positive moves content downward
        (toward higher row indices).
    dx : int
        Shift along axis 1 (columns).  Positive moves content rightward
        (toward higher column indices).

    Returns
    -------
    ndarray
        Shifted image with wrapped edges replaced by NaN.
    """
    out = np.roll(image, shift=(dy, dx), axis=(0, 1)).astype(float, copy=True)

    # Blank the rows that were wrapped in.
    if dy > 0:
        out[:dy, :] = np.nan   # top dy rows are garbage from the bottom wrap
    elif dy < 0:
        out[dy:, :] = np.nan   # bottom |dy| rows are garbage from the top wrap

    # Blank the columns that were wrapped in.
    if dx > 0:
        out[:, :dx] = np.nan   # left dx cols are garbage from the right wrap
    elif dx < 0:
        out[:, dx:] = np.nan   # right |dx| cols are garbage from the left wrap

    return out


def subtract_xy_median_pattern(image: np.ndarray) -> np.ndarray:
    """Remove vertical/horizontal striping using median profiles in x and y."""

    out = np.asarray(image, dtype=float).copy()

    # Subtract vertical pattern: one median value per column.
    col_profile = np.nanmedian(out, axis=0)
    out -= col_profile[np.newaxis, :]

    # Subtract horizontal pattern: one median value per row.
    row_profile = np.nanmedian(out, axis=1)
    out -= row_profile[:, np.newaxis]

    return out


def make_merged_product(
    object_name: str,
    selected_files: List[str],
    output_folder_merged: str,
) -> Optional[str]:
    """Build registered cube and median FOV for one object."""

    if not selected_files:
        print(f"  No files selected for object '{object_name}'.")
        return None

    print(f"  Building cube for '{object_name}' from {len(selected_files)} file(s)")

    guiding_images: List[np.ndarray] = []
    residual_images: List[np.ndarray] = []
    centers: List[Tuple[float, float]] = []
    used_files: List[str] = []

    shape_ref: Optional[Tuple[int, int]] = None

    for filepath in selected_files:
        try:
            guiding_image, residual_image, xcen, ycen, _header = read_analysis_images(filepath)
        except Exception as exc:
            print(f"  Skip unreadable file: {filepath} ({exc})")
            continue

        if shape_ref is None:
            shape_ref = guiding_image.shape
        if guiding_image.shape != shape_ref:
            print(
                f"  Skip shape mismatch: {filepath} has {guiding_image.shape}, expected {shape_ref}"
            )
            continue

        guiding_images.append(np.asarray(guiding_image, dtype=float))
        residual_images.append(np.asarray(residual_image, dtype=float))
        centers.append((xcen, ycen))
        used_files.append(filepath)
        print(f"  Keep: {os.path.basename(filepath)} | XCEN={xcen:.2f}, YCEN={ycen:.2f}")

    if not guiding_images:
        print(f"  No valid images left for '{object_name}' after filtering.")
        return None

    centers_arr = np.asarray(centers, dtype=float)
    x_med = float(np.nanmedian(centers_arr[:, 0]))
    y_med = float(np.nanmedian(centers_arr[:, 1]))
    print(f"  Median centroid target: ({x_med:.2f}, {y_med:.2f})")

    # --- Registration ---
    # Use the XCEN/YCEN values that were optimized by get_props_guiding.py.
    # The shift needed to align frame i to the median position is:
    #   dx = x_med - xcen_i   (positive → shift content rightward)
    #   dy = y_med - ycen_i   (positive → shift content downward)
    # np.roll(arr, +k, axis=1) moves each column value to column+k,
    # so a star at xcen_i ends up at xcen_i + dx = x_med.  ✓

    n_frames = len(guiding_images)
    registered_guiding_cube = np.full((n_frames, shape_ref[0], shape_ref[1]), np.nan, dtype=float)
    registered_residual_cube = np.full((n_frames, shape_ref[0], shape_ref[1]), np.nan, dtype=float)

    shifts: List[Tuple[int, int]] = []
    for idx, ((guiding_image, residual_image), (xcen, ycen)) in enumerate(
        zip(zip(guiding_images, residual_images), centers)
    ):
        dx = int(np.rint(x_med - xcen))
        dy = int(np.rint(y_med - ycen))
        shifts.append((dy, dx))
        registered_guiding_cube[idx] = integer_shift_with_nan_padding(guiding_image, dy=dy, dx=dx)
        registered_residual_cube[idx] = integer_shift_with_nan_padding(residual_image, dy=dy, dx=dx)
        print(
            f"  Register frame {idx + 1}/{n_frames}: "
            f"XCEN={xcen:.2f}, YCEN={ycen:.2f}  ->  shift(dy={dy}, dx={dx})"
        )

    median_guiding = np.nanmedian(registered_guiding_cube, axis=0)
    median_residual = np.nanmedian(registered_residual_cube, axis=0)
    median_residual = subtract_xy_median_pattern(median_residual)
    print(f"  Median stack done: {n_frames} frames, image shape {shape_ref}")
    print("  Applied x/y median-profile subtraction to merged RESIDUAL")

    os.makedirs(output_folder_merged, exist_ok=True)
    object_tag = "_".join(object_name.strip().split())
    output_filename = f"{object_tag}_guiding_median_fov.fits"
    output_path = os.path.join(output_folder_merged, output_filename)

    hdr = fits.Header()
    hdr["OBJECT"] = object_name
    hdr["NINPUT"] = (len(used_files), "Number of frames combined")
    hdr["XCENMED"] = (x_med, "Median XCEN used as registration target")
    hdr["YCENMED"] = (y_med, "Median YCEN used as registration target")
    hdr["COMMENT"] = "Aligned with integer np.roll; wrapped edges replaced by NaN"

    primary_hdu = fits.PrimaryHDU(header=hdr)
    # Extension 1: median of registered GUIDING images (pre-radial-subtraction)
    guiding_hdu = fits.ImageHDU(median_guiding, name="GUIDING")
    # Extension 2: median of registered RESIDUAL images (post-radial-subtraction)
    residual_hdu = fits.ImageHDU(median_residual, name="RESIDUAL")
    # Extension 3: full registered cube of GUIDING frames
    cube_hdu = fits.ImageHDU(registered_guiding_cube, name="REGISTERED_CUBE")

    col_file = fits.Column(name="FILE", format="A256", array=np.asarray(used_files, dtype="S256"))
    col_x = fits.Column(name="XCEN", format="D", array=np.asarray([c[0] for c in centers]))
    col_y = fits.Column(name="YCEN", format="D", array=np.asarray([c[1] for c in centers]))
    col_dy = fits.Column(name="SHIFT_DY", format="J", array=np.asarray([s[0] for s in shifts], dtype=np.int32))
    col_dx = fits.Column(name="SHIFT_DX", format="J", array=np.asarray([s[1] for s in shifts], dtype=np.int32))
    provenance_hdu = fits.BinTableHDU.from_columns(
        [col_file, col_x, col_y, col_dy, col_dx],
        name="PROVENANCE",
    )

    fits.HDUList([primary_hdu, guiding_hdu, residual_hdu, cube_hdu, provenance_hdu]).writeto(
        output_path,
        overwrite=True,
    )

    print(f"  Saved merged FITS: {output_path}")
    print(f"  Merge summary for '{object_name}': {len(used_files)} frame(s) combined")

    return output_path


def group_files_by_object(filepaths: List[str]) -> Dict[str, List[str]]:
    """Group files by normalized object name from primary headers."""

    grouped: Dict[str, List[str]] = {}
    total = len(filepaths)
    print(f"Scanning headers for {total} candidate file(s)...")

    for idx, filepath in enumerate(filepaths, start=1):
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

        if idx <= 5:
            print(
                f"  Header {idx}/{total}: {os.path.basename(filepath)} -> "
                f"OBJECT='{object_name}'"
            )

        if idx == 1 or idx == total or idx % 100 == 0:
            print(f"  Parsed {idx}/{total} headers")

    print(f"Header scan done. Found {len(grouped)} unique object name(s).")

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

    # Normalize path so trailing separators do not turn '_merged' into a subfolder.
    out_base = os.path.normpath(os.path.expanduser(out_base))
    output_folder_merged = f"{out_base}_merged"

    filepaths = get_fits_candidates(base=out_base, wildcard=args.scan_pattern)
    if not filepaths:
        raise SystemExit(
            f"No candidate FITS files found in output folder '{out_base}' "
            f"with pattern '{args.scan_pattern}'."
        )

    print(
        f"Discovered {len(filepaths)} candidate file(s) in '{out_base}' "
        f"with pattern '{args.scan_pattern}'"
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
        if selected:
            print(f"  First match: {os.path.basename(selected[0])}")

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
