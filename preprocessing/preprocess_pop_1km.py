from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio
from numcodecs import Blosc
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
import zarr


ROOT = Path(__file__).resolve().parents[1]
IN_DIR_DEFAULT = ROOT / "1007_pop" / "preprocessed"
OUT_DIR_DEFAULT = ROOT / "3004_pop"
VAL_DIR_NAME = "validation_tif"

# Canonical 1km domain
XMIN, YMIN = 700_000, 1_400_000
NX, NY = 610, 700
RES_1KM = 1_000
YMAX = YMIN + NY * RES_1KM
CRS_5179 = CRS.from_epsg(5179)
DST_TRANSFORM = from_origin(XMIN, YMAX, RES_1KM, RES_1KM)
GEOTRANSFORM = f"{XMIN} {RES_1KM} 0 {YMAX} 0 -{RES_1KM}"

COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _read_source(path: Path) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, CRS]:
    with rasterio.open(path) as ds:
        src = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        src_crs = ds.crs
        src_transform = ds.transform

    if src_crs is None:
        raise RuntimeError(f"Source CRS is missing: {path}")

    valid = np.isfinite(src)
    if nodata is not None and np.isfinite(nodata):
        valid &= src != np.float32(nodata)

    src_filled = np.where(valid, src, 0.0).astype(np.float32)
    src_valid = valid.astype(np.float32)
    return src_filled, src_valid, src_transform, src_crs


def _aggregate_to_1km(
    src_val: np.ndarray,
    src_valid: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    dst_sum = np.zeros((NY, NX), dtype=np.float64)
    dst_wgt = np.zeros((NY, NX), dtype=np.float64)

    reproject(
        source=src_val,
        destination=dst_sum,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=DST_TRANSFORM,
        dst_crs=CRS_5179,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.sum,
    )

    reproject(
        source=src_valid,
        destination=dst_wgt,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=DST_TRANSFORM,
        dst_crs=CRS_5179,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.sum,
    )

    out = dst_sum.astype(np.float32)
    out[dst_wgt <= 0.0] = np.nan
    return out, dst_wgt.astype(np.float32)


def _write_validation_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        width=NX,
        height=NY,
        count=1,
        dtype="float32",
        crs=CRS_5179,
        transform=DST_TRANSFORM,
        nodata=-9999.0,
        compress="LZW",
    ) as ds:
        ds.write(out, 1)
        ds.set_band_description(1, "population_total_1km_sum")


def _write_zarr(path: Path, year: int, arr_1km: np.ndarray, overwrite: bool) -> zarr.Array:
    if path.exists() and overwrite:
        shutil.rmtree(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path} (use --overwrite)")

    z = zarr.open(
        str(path),
        mode="w",
        shape=(1, NY, NX),
        chunks=(1, NY, NX),
        dtype="float32",
        fill_value=np.nan,
        compressor=COMPRESSOR,
    )
    z[0, :, :] = arr_1km

    z.attrs.update(
        {
            "description": "Yearly population total aggregated to canonical 1km grid",
            "variable": "population_total",
            "source": f"1007_pop/preprocessed/pop_100m_{year}.tif",
            "source_stat_code": "to_in_001",
            "aggregation_method": "area_weighted_sum_resampling",
            "crs": "EPSG:5179",
            "spatial_ref": CRS_5179.to_wkt(),
            "crs_wkt": CRS_5179.to_wkt(),
            "GeoTransform": GEOTRANSFORM,
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],
            "time_values": [f"{year}-01-01T00:00:00Z"],
            "year": year,
            "xmin": XMIN,
            "ymin": YMIN,
            "ymax": YMAX,
            "nx": NX,
            "ny": NY,
            "res_m": RES_1KM,
            "origin": "NW",
            "units": "persons",
            "note": "NaN indicates out-of-domain cells.",
        }
    )
    return z


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert yearly 100m population GeoTIFF into canonical 1km Zarr")
    ap.add_argument("--year", type=int, required=True, help="Target year (e.g. 2015)")
    ap.add_argument("--in-dir", default=str(IN_DIR_DEFAULT), help="Input directory for pop_100m_{year}.tif")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT), help="Output directory for Zarr/validation files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output zarr")
    ap.add_argument("--no-validation-tif", action="store_true", help="Skip validation GeoTIFF export")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    val_dir = out_dir / VAL_DIR_NAME

    src_tif = in_dir / f"pop_100m_{args.year}.tif"
    out_zarr = out_dir / f"pop_{args.year}.zarr"
    val_tif = val_dir / f"pop_{args.year}_1km_sum_qgis.tif"

    if not src_tif.exists():
        print(f"[ERROR] missing input GeoTIFF: {src_tif}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 2  Population 100m -> 1km Zarr")
    print("=" * 72)
    print(f"  year      : {args.year}")
    print(f"  input     : {src_tif}")
    print(f"  output    : {out_zarr}")
    print(f"  grid      : {NX}x{NY}  res={RES_1KM}m  CRS=EPSG:5179  origin=NW")

    print("\n[1/4] read source")
    src_val, src_valid, src_transform, src_crs = _read_source(src_tif)
    in_sum = float(np.sum(src_val, dtype=np.float64))
    in_valid = int(np.count_nonzero(src_valid > 0))
    print(f"  source CRS={src_crs}")
    print(f"  source sum={in_sum:.3f} valid_cells={in_valid:,}")

    print("\n[2/4] area-weighted sum aggregation")
    arr_1km, wgt_1km = _aggregate_to_1km(src_val, src_valid, src_transform, src_crs)
    out_sum = float(np.nansum(arr_1km, dtype=np.float64))
    out_valid = int(np.count_nonzero(np.isfinite(arr_1km)))
    rel = (out_sum - in_sum) / max(1.0, abs(in_sum))
    print(f"  output sum={out_sum:.3f} valid_cells={out_valid:,}")
    print(f"  sum diff={out_sum - in_sum:.3f} ({rel:.6%})")

    max_w = float(np.nanmax(wgt_1km)) if np.isfinite(wgt_1km).any() else 0.0
    min_pos_w = float(np.nanmin(wgt_1km[wgt_1km > 0])) if np.any(wgt_1km > 0) else 0.0
    print(f"  overlap weight range (positive): min={min_pos_w:.6f}, max={max_w:.6f}")

    print("\n[3/4] write zarr")
    z = _write_zarr(out_zarr, args.year, arr_1km, overwrite=args.overwrite)
    print(f"  zarr shape={z.shape} chunks={z.chunks}")

    print("\n[4/4] write validation tif")
    if args.no_validation_tif:
        print("  skipped (--no-validation-tif)")
    else:
        _write_validation_tif(val_tif, arr_1km)
        print(f"  wrote {val_tif}")

    if not math.isfinite(out_sum):
        print("[ERROR] output sum is not finite")
        return 2

    print("\nDone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
