from __future__ import annotations

import argparse
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from numcodecs import Blosc, Zlib
from pyproj import Transformer
from rasterio.crs import CRS


UTC = timezone.utc
MONTH_RE = re.compile(r"^(\d{6})(?:-(\d{6}))?$")
DAY_TIF_RE = re.compile(r"^era5_land_(\d{8})_h00_23\.tif$", re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[1]
LAND_ROOT = ROOT / "1002_era5-land" / "raw"
ERA5_ROOT = ROOT / "1001_era5" / "NCAR_Curated_AWS" / "raw"
OUT_ROOT_DEFAULT = ROOT / "2001_era5"
CANONICAL_GRID_ZARR = ROOT / "3001_dem" / "dem.zarr"

ZARR_VERSION = 2
CRS_5179 = CRS.from_epsg(5179)
CRS_5179_WKT = CRS_5179.to_wkt()
CRS_4326_WKT = CRS.from_epsg(4326).to_wkt()
TF_5179_TO_4326 = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
TF_4326_TO_5179 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

LAND_TIF_BAND_ORDER = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
]

LAND_BAND_VARS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_pressure",
]

ACCUM_LAND_VARS: set[str] = set()

OUT_CHANNELS = [
    "t2m",
    "rh2m",
    "u10",
    "v10",
    "psfc",
    "pblh",
    "tp_1h",
    "ssrd_1h",
    "strd_1h",
    "tcc",
    "lcc",
    "mcc",
    "u925",
    "v925",
    "t925",
    "rh925",
    "w925",
    "u850",
    "v850",
    "t850",
    "rh850",
    "w850",
    "u700",
    "v700",
    "t700",
    "rh700",
    "ws10",
    "ws925",
    "ws850",
    "shear_925_10m",
    "ventilation_index",
]

ERA5_DIRECT_VARS = [
    "pblh",
    "tcc",
    "lcc",
    "mcc",
    "u925",
    "v925",
    "t925",
    "rh925",
    "w925",
    "u850",
    "v850",
    "t850",
    "rh850",
    "w850",
    "u700",
    "v700",
    "t700",
    "rh700",
    "ws925",
    "ws850",
    "shear_925_10m",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build 2001 monthly 31-channel train-domain zarr from 1001+1002")
    ap.add_argument("--month", required=True, help="YYYYMM or YYYYMM-YYYYMM")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing month zarr")
    ap.add_argument("--engine", choices=["cuda", "cpu"], default="cuda", help="Interpolation engine")
    ap.add_argument("--days", default="", help="Optional day subset csv, e.g. 01,02")
    ap.add_argument(
        "--compressor",
        choices=["blosc-zstd", "blosc-lz4", "zlib", "none"],
        default="blosc-zstd",
        help="Output chunk compressor",
    )
    ap.add_argument("--clevel", type=int, default=3, help="Compression level (zlib:0-9, blosc:0-9)")
    ap.add_argument(
        "--shuffle",
        choices=["bitshuffle", "shuffle", "noshuffle"],
        default="bitshuffle",
        help="Shuffle mode for Blosc compressors",
    )
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT), help="Output root directory")
    ap.add_argument(
        "--output-layout",
        choices=["compact", "full5179"],
        default="compact",
        help="compact: keep 63x70 lat/lon and add 5179 aux coords, full5179: write full 3001_dem grid",
    )
    return ap.parse_args()


def build_compressor(codec: str, clevel: int, shuffle: str):
    level = max(0, min(9, int(clevel)))
    if codec == "none":
        return None
    if codec == "zlib":
        return Zlib(level=level)

    shuffle_map = {
        "bitshuffle": Blosc.BITSHUFFLE,
        "shuffle": Blosc.SHUFFLE,
        "noshuffle": Blosc.NOSHUFFLE,
    }
    cname = "zstd" if codec == "blosc-zstd" else "lz4"
    return Blosc(cname=cname, clevel=level, shuffle=shuffle_map[shuffle])


def compressor_log_label(codec: str, clevel: int, shuffle: str) -> str:
    level = max(0, min(9, int(clevel)))
    if codec == "none":
        return "none"
    if codec == "zlib":
        return f"zlib(level={level})"
    cname = "zstd" if codec == "blosc-zstd" else "lz4"
    return f"blosc(cname={cname}, clevel={level}, shuffle={shuffle})"


def parse_yyyymm(token: str) -> datetime:
    year = int(token[:4])
    month = int(token[4:6])
    if month < 1 or month > 12:
        raise ValueError(f"invalid month: {token}")
    return datetime(year, month, 1, tzinfo=UTC)


def parse_month_spec(spec: str) -> tuple[datetime, datetime]:
    m = MONTH_RE.match(spec)
    if not m:
        raise ValueError(f"invalid --month spec: {spec}")
    start = parse_yyyymm(m.group(1))
    end = parse_yyyymm(m.group(2)) if m.group(2) else start
    if end < start:
        raise ValueError(f"invalid month range: {spec}")
    return start, end


def iter_months(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m")
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=UTC)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=UTC)


def resolve_days(yyyymm: str, raw_days: str) -> list[str]:
    month_dir = LAND_ROOT / yyyymm
    picked: list[str] = []
    for p in sorted(month_dir.iterdir()):
        if not p.is_file():
            continue
        m = DAY_TIF_RE.match(p.name)
        if m:
            picked.append(m.group(1)[6:8])
    if not picked:
        raise RuntimeError(f"no ERA5-Land tif files found for {yyyymm}")

    if not raw_days.strip():
        return picked

    # Accept both comma-list and ranges, and normalize 1/01 style inputs.
    requested_set: set[str] = set()
    for token in [x.strip() for x in raw_days.split(",") if x.strip()]:
        if "-" in token:
            parts = [p.strip() for p in token.split("-", 1)]
            if len(parts) != 2 or (not parts[0].isdigit()) or (not parts[1].isdigit()):
                raise RuntimeError(f"invalid --days token: {token!r} (use e.g. 01,02 or 1-5)")
            s = int(parts[0])
            e = int(parts[1])
            if s < 1 or s > 31 or e < 1 or e > 31 or e < s:
                raise RuntimeError(f"invalid --days range: {token!r}")
            for d in range(s, e + 1):
                requested_set.add(f"{d:02d}")
        else:
            if not token.isdigit():
                raise RuntimeError(f"invalid --days token: {token!r} (use e.g. 01,02 or 1-5)")
            d = int(token)
            if d < 1 or d > 31:
                raise RuntimeError(f"invalid --days value: {token!r}")
            requested_set.add(f"{d:02d}")

    requested = sorted(requested_set)
    missing = [d for d in requested if d not in picked]
    if missing:
        raise RuntimeError(f"requested days missing in {yyyymm}: {missing}")
    return requested


def load_train_grid() -> dict[str, np.ndarray | str | int | float]:
    arr = zarr.open_array(str(CANONICAL_GRID_ZARR), mode="r")
    gt = str(arr.attrs["GeoTransform"]).split()
    xmin = float(gt[0])
    res = float(gt[1])
    ymax = float(gt[3])
    ny, nx = arr.shape
    x = (xmin + (np.arange(nx, dtype=np.float64) + 0.5) * res).astype(np.float32)
    y = (ymax - (np.arange(ny, dtype=np.float64) + 0.5) * res).astype(np.float32)
    xx, yy = np.meshgrid(x.astype(np.float64), y.astype(np.float64))
    lon, lat = TF_5179_TO_4326.transform(xx, yy)
    return {
        "xmin": xmin,
        "res": res,
        "ymax": ymax,
        "nx": int(nx),
        "ny": int(ny),
        "x": x,
        "y": y,
        "lon2d": lon.astype(np.float32),
        "lat2d": lat.astype(np.float32),
        "crs": str(arr.attrs.get("crs", "EPSG:5179")),
        "crs_wkt": str(arr.attrs.get("crs_wkt", CRS_5179_WKT)),
        "geotransform": str(arr.attrs["GeoTransform"]),
    }


def mask_invalid(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    invalid = (~np.isfinite(out)) | (out == 9999.0) | (out == -9999.0) | (np.abs(out) > 1.0e19)
    if np.any(invalid):
        out = out.copy()
        out[invalid] = np.nan
    return out


def dewpoint_to_rh(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
    tc = np.clip(t2m.astype(np.float64) - 273.15, -80.0, 60.0)
    tdc = np.clip(d2m.astype(np.float64) - 273.15, -100.0, 60.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        es = np.exp(17.625 * tc / (243.04 + tc))
        e = np.exp(17.625 * tdc / (243.04 + tdc))
        rh = 100.0 * e / np.maximum(es, 1.0e-12)
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


def land_band_index(var_name: str, hour_utc: int) -> int:
    return hour_utc * len(LAND_TIF_BAND_ORDER) + LAND_TIF_BAND_ORDER.index(var_name) + 1


def crop_slice(values: np.ndarray, coord: np.ndarray, vmin: float, vmax: float, pad: int = 1) -> tuple[slice, np.ndarray]:
    mask = (coord >= vmin) & (coord <= vmax)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError("target bbox does not intersect source grid")
    start = max(0, int(idx[0]) - pad)
    end = min(coord.size, int(idx[-1]) + pad + 1)
    return slice(start, end), coord[start:end]


def source_target_grid(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat2d: np.ndarray,
    tgt_lon2d: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    src_lat = np.asarray(src_lat, dtype=np.float32)
    src_lon = np.asarray(src_lon, dtype=np.float32)
    top = float(src_lat[0])
    bottom = float(src_lat[-1])
    left = float(src_lon[0])
    right = float(src_lon[-1])

    gx = (2.0 * (tgt_lon2d.astype(np.float32) - left) / (right - left)) - 1.0
    gy = (2.0 * (top - tgt_lat2d.astype(np.float32)) / (top - bottom)) - 1.0
    grid = np.stack([gx, gy], axis=-1)
    return torch.from_numpy(grid).to(device=device, dtype=torch.float32).unsqueeze(0)


def interp_regular_grid(
    src_values: np.ndarray,
    grid: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    src = torch.from_numpy(src_values.astype(np.float32, copy=False)).to(device=device)
    if src.ndim == 2:
        src = src.unsqueeze(0)

    valid = torch.isfinite(src)
    src_filled = torch.nan_to_num(src, nan=0.0)
    inp = src_filled.unsqueeze(1)
    wgt = valid.float().unsqueeze(1)
    batch_grid = grid.expand(src.shape[0], -1, -1, -1)

    sampled_val = F.grid_sample(inp, batch_grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)
    sampled_wgt = F.grid_sample(wgt, batch_grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)
    out = torch.where(sampled_wgt > 1.0e-6, sampled_val / sampled_wgt.clamp_min(1.0e-6), torch.full_like(sampled_val, float("nan")))
    return out.detach().cpu().numpy().astype(np.float32)


def load_prev_day_last_accum(day_stamp: str, var_name: str, land_lat_sl: slice, land_lon_sl: slice) -> np.ndarray | None:
    day_dt = datetime.strptime(day_stamp, "%Y%m%d").replace(tzinfo=UTC)
    prev_dt = day_dt - timedelta(days=1)
    prev_month = prev_dt.strftime("%Y%m")
    prev_stamp = prev_dt.strftime("%Y%m%d")
    prev_tif = LAND_ROOT / prev_month / f"era5_land_{prev_stamp}_h00_23.tif"
    if not prev_tif.exists():
        return None

    with rasterio.open(prev_tif) as src:
        arr = src.read(land_band_index(var_name, 23))[land_lat_sl, land_lon_sl]
        return mask_invalid(arr)


def deaccumulate_stack(stack: np.ndarray, prev_last: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    out = np.full_like(stack, np.nan, dtype=np.float32)
    if prev_last is None:
        out[0] = stack[0]
    else:
        d0 = stack[0] - prev_last
        reset = ~np.isfinite(d0) | (d0 < 0.0)
        out[0] = np.where(reset, stack[0], d0)

    for h in range(1, stack.shape[0]):
        dh = stack[h] - stack[h - 1]
        reset = ~np.isfinite(dh) | (dh < 0.0)
        out[h] = np.where(reset, stack[h], dh)

    out = np.where(np.isfinite(out), np.maximum(out, 0.0), np.nan).astype(np.float32)
    return out, stack[-1].astype(np.float32)


def build_output_group_full(
    path: Path,
    times: pd.DatetimeIndex,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    geotransform: str,
    compressor,
) -> zarr.Group:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ny = int(np.asarray(y_coords).shape[0])
    nx = int(np.asarray(x_coords).shape[0])

    g = zarr.open_group(str(path), mode="w", zarr_version=ZARR_VERSION)
    g.attrs["title"] = "2001 ERA5 31-channel train-domain monthly package"
    g.attrs["source"] = "1002_era5-land(raw tif) + 1001_era5(monthly zarr)"
    g.attrs["Conventions"] = "CF-1.8"
    g.attrs["crs"] = "EPSG:5179"
    g.attrs["crs_wkt"] = CRS_5179_WKT
    g.attrs["spatial_ref"] = CRS_5179_WKT
    g.attrs["GeoTransform"] = geotransform
    g.attrs["width"] = nx
    g.attrs["height"] = ny
    g.attrs["channels"] = OUT_CHANNELS

    tvals = np.array([t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times], dtype="U20")
    g.create_dataset("time", shape=(len(times),), dtype="U20", chunks=(min(744, max(24, len(times))),), overwrite=True)[:] = tvals
    g["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    g["time"].attrs["standard_name"] = "time"
    g["time"].attrs["long_name"] = "time"

    x = np.asarray(x_coords, dtype=np.float32)
    y = np.asarray(y_coords, dtype=np.float32)
    g.create_dataset("x", shape=x.shape, dtype="f4", chunks=x.shape, overwrite=True)[:] = x
    g["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]
    g["x"].attrs["standard_name"] = "projection_x_coordinate"
    g["x"].attrs["long_name"] = "x coordinate of projection"
    g["x"].attrs["units"] = "m"

    g.create_dataset("y", shape=y.shape, dtype="f4", chunks=y.shape, overwrite=True)[:] = y
    g["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    g["y"].attrs["standard_name"] = "projection_y_coordinate"
    g["y"].attrs["long_name"] = "y coordinate of projection"
    g["y"].attrs["units"] = "m"

    g.create_dataset("longitude", shape=lon2d.shape, dtype="f4", chunks=lon2d.shape, overwrite=True)[:] = np.asarray(lon2d, dtype=np.float32)
    g["longitude"].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    g["longitude"].attrs["standard_name"] = "longitude"
    g["longitude"].attrs["long_name"] = "longitude"
    g["longitude"].attrs["units"] = "degrees_east"

    g.create_dataset("latitude", shape=lat2d.shape, dtype="f4", chunks=lat2d.shape, overwrite=True)[:] = np.asarray(lat2d, dtype=np.float32)
    g["latitude"].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    g["latitude"].attrs["standard_name"] = "latitude"
    g["latitude"].attrs["long_name"] = "latitude"
    g["latitude"].attrs["units"] = "degrees_north"

    g.create_dataset("spatial_ref", shape=(), dtype="i4", chunks=(), overwrite=True)[()] = 0
    g["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []
    g["spatial_ref"].attrs["spatial_ref"] = CRS_5179_WKT
    g["spatial_ref"].attrs["crs_wkt"] = CRS_5179_WKT
    g["spatial_ref"].attrs["grid_mapping_name"] = "transverse_mercator"
    g["spatial_ref"].attrs["epsg_code"] = "EPSG:5179"

    for name in OUT_CHANNELS:
        arr = g.create_dataset(
            name,
            shape=(len(times), ny, nx),
            dtype="f4",
            chunks=(24, ny, nx),
            compressor=compressor,
            fill_value=np.nan,
            overwrite=True,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
        arr.attrs["coordinates"] = "latitude longitude"
        arr.attrs["grid_mapping"] = "spatial_ref"
        arr.attrs["spatial_ref"] = CRS_5179_WKT
        arr.attrs["crs_wkt"] = CRS_5179_WKT

    return g


def build_output_group_compact(
    path: Path,
    times: pd.DatetimeIndex,
    land_lat: np.ndarray,
    land_lon: np.ndarray,
    x_5179_2d: np.ndarray,
    y_5179_2d: np.ndarray,
    compressor,
) -> zarr.Group:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ny = int(len(land_lat))
    nx = int(len(land_lon))

    g = zarr.open_group(str(path), mode="w", zarr_version=ZARR_VERSION)
    g.attrs["title"] = "2001 ERA5 31-channel train-domain monthly package"
    g.attrs["source"] = "1002_era5-land(raw tif) + 1001_era5(monthly zarr)"
    g.attrs["Conventions"] = "CF-1.8"
    g.attrs["coord_layout"] = "compact_latlon_with_5179_aux"
    g.attrs["crs"] = "EPSG:4326"
    g.attrs["crs_wkt"] = CRS_4326_WKT
    g.attrs["aux_crs"] = "EPSG:5179"
    g.attrs["aux_crs_wkt"] = CRS_5179_WKT
    g.attrs["width"] = nx
    g.attrs["height"] = ny
    g.attrs["channels"] = OUT_CHANNELS

    tvals = np.array([t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times], dtype="U20")
    g.create_dataset("time", shape=(len(times),), dtype="U20", chunks=(min(744, max(24, len(times))),), overwrite=True)[:] = tvals
    g["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    g["time"].attrs["standard_name"] = "time"
    g["time"].attrs["long_name"] = "time"

    lon = np.asarray(land_lon, dtype=np.float32)
    lat = np.asarray(land_lat, dtype=np.float32)
    g.create_dataset("longitude", shape=lon.shape, dtype="f4", chunks=lon.shape, overwrite=True)[:] = lon
    g["longitude"].attrs["_ARRAY_DIMENSIONS"] = ["longitude"]
    g["longitude"].attrs["standard_name"] = "longitude"
    g["longitude"].attrs["long_name"] = "longitude"
    g["longitude"].attrs["units"] = "degrees_east"

    g.create_dataset("latitude", shape=lat.shape, dtype="f4", chunks=lat.shape, overwrite=True)[:] = lat
    g["latitude"].attrs["_ARRAY_DIMENSIONS"] = ["latitude"]
    g["latitude"].attrs["standard_name"] = "latitude"
    g["latitude"].attrs["long_name"] = "latitude"
    g["latitude"].attrs["units"] = "degrees_north"

    g.create_dataset("x_5179", shape=x_5179_2d.shape, dtype="f4", chunks=x_5179_2d.shape, overwrite=True)[:] = np.asarray(x_5179_2d, dtype=np.float32)
    g["x_5179"].attrs["_ARRAY_DIMENSIONS"] = ["latitude", "longitude"]
    g["x_5179"].attrs["long_name"] = "EPSG:5179 projected x"
    g["x_5179"].attrs["units"] = "m"

    g.create_dataset("y_5179", shape=y_5179_2d.shape, dtype="f4", chunks=y_5179_2d.shape, overwrite=True)[:] = np.asarray(y_5179_2d, dtype=np.float32)
    g["y_5179"].attrs["_ARRAY_DIMENSIONS"] = ["latitude", "longitude"]
    g["y_5179"].attrs["long_name"] = "EPSG:5179 projected y"
    g["y_5179"].attrs["units"] = "m"

    g.create_dataset("spatial_ref", shape=(), dtype="i4", chunks=(), overwrite=True)[()] = 0
    g["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []
    g["spatial_ref"].attrs["spatial_ref"] = CRS_4326_WKT
    g["spatial_ref"].attrs["crs_wkt"] = CRS_4326_WKT
    g["spatial_ref"].attrs["grid_mapping_name"] = "latitude_longitude"

    for name in OUT_CHANNELS:
        arr = g.create_dataset(
            name,
            shape=(len(times), ny, nx),
            dtype="f4",
            chunks=(24, ny, nx),
            compressor=compressor,
            fill_value=np.nan,
            overwrite=True,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
        arr.attrs["coordinates"] = "latitude longitude x_5179 y_5179"
        arr.attrs["grid_mapping"] = "spatial_ref"
        arr.attrs["spatial_ref"] = CRS_4326_WKT
        arr.attrs["crs_wkt"] = CRS_4326_WKT

    return g


def process_month(yyyymm: str, overwrite: bool, engine: str, day_subset: str, out_root: Path, compressor, output_layout: str) -> Path:
    if engine == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--engine cuda requested but CUDA is not available")

    month_t0 = time.perf_counter()
    device = torch.device("cuda" if engine == "cuda" else "cpu")
    grid = load_train_grid()
    train_lat2d = np.asarray(grid["lat2d"], dtype=np.float32)
    train_lon2d = np.asarray(grid["lon2d"], dtype=np.float32)
    train_lat_min = float(np.nanmin(train_lat2d))
    train_lat_max = float(np.nanmax(train_lat2d))
    train_lon_min = float(np.nanmin(train_lon2d))
    train_lon_max = float(np.nanmax(train_lon2d))

    month_dir = LAND_ROOT / yyyymm
    days = resolve_days(yyyymm, day_subset)
    tif_paths = [month_dir / f"era5_land_{yyyymm}{dd}_h00_23.tif" for dd in days]

    era5_path = ERA5_ROOT / yyyymm / f"era5_ncar_curated_eastasia_{yyyymm}.zarr"
    ds_era5 = xr.open_zarr(era5_path)
    try:
        era5_times = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds_era5["time"].values), utc=True))
        full_start = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:6]}-01T00:00:00Z")
        expected_full = pd.date_range(full_start, full_start + pd.offsets.MonthBegin(1), freq="h", inclusive="left")
        if len(era5_times) != len(expected_full) or not era5_times.equals(expected_full):
            raise RuntimeError(f"ERA5 time axis mismatch for {yyyymm}")

        out_times: list[pd.Timestamp] = []
        for dd in days:
            base = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:6]}-{dd}T00:00:00Z")
            out_times.extend(pd.date_range(base, periods=24, freq="h"))
        out_times_idx = pd.DatetimeIndex(out_times)

        out_dir = out_root / yyyymm
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"era5_31ch_train_domain_{yyyymm}.zarr"
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {yyyymm}: {out_path}")
            return out_path

        first_tif = tif_paths[0]
        with rasterio.open(first_tif) as src0:
            if str(src0.crs).upper() != "EPSG:4326":
                raise RuntimeError(f"unexpected ERA5-Land tif CRS: {src0.crs}")
            land_lon_full = (src0.transform.c + src0.transform.a * (np.arange(src0.width, dtype=np.float64) + 0.5)).astype(np.float32)
            land_lat_full = (src0.transform.f + src0.transform.e * (np.arange(src0.height, dtype=np.float64) + 0.5)).astype(np.float32)

        land_lat_sl, land_lat = crop_slice(land_lat_full, land_lat_full, train_lat_min, train_lat_max, pad=0)
        land_lon_sl, land_lon = crop_slice(land_lon_full, land_lon_full, train_lon_min, train_lon_max, pad=0)
        land_lon2d, land_lat2d = np.meshgrid(land_lon.astype(np.float32), land_lat.astype(np.float32))

        out_x2d, out_y2d = TF_4326_TO_5179.transform(land_lon2d, land_lat2d)
        out_x2d = np.asarray(out_x2d, dtype=np.float32)
        out_y2d = np.asarray(out_y2d, dtype=np.float32)

        era5_lat_full = np.asarray(ds_era5["latitude"].values, dtype=np.float32)
        era5_lon_full = np.asarray(ds_era5["longitude"].values, dtype=np.float32)
        era5_lat_sl, era5_lat = crop_slice(era5_lat_full, era5_lat_full, train_lat_min, train_lat_max, pad=2)
        era5_lon_sl, era5_lon = crop_slice(era5_lon_full, era5_lon_full, train_lon_min, train_lon_max, pad=2)

        era5_to_land_grid = source_target_grid(era5_lat, era5_lon, land_lat2d, land_lon2d, device)
        if output_layout == "full5179":
            out_x = np.asarray(grid["x"], dtype=np.float32)
            out_y = np.asarray(grid["y"], dtype=np.float32)
            out_lat2d = np.asarray(grid["lat2d"], dtype=np.float32)
            out_lon2d = np.asarray(grid["lon2d"], dtype=np.float32)
            land_to_target_grid = source_target_grid(land_lat, land_lon, out_lat2d, out_lon2d, device)
            era5_to_target_grid = source_target_grid(era5_lat, era5_lon, out_lat2d, out_lon2d, device)
            g = build_output_group_full(out_path, out_times_idx, out_x, out_y, out_lat2d, out_lon2d, grid["geotransform"], compressor)
            grid_msg = f"{grid['ny']}x{grid['nx']}"
        else:
            land_to_target_grid = None
            era5_to_target_grid = None
            g = build_output_group_compact(out_path, out_times_idx, land_lat, land_lon, out_x2d, out_y2d, compressor)
            grid_msg = f"{len(land_lat)}x{len(land_lon)}"
        out_arrays = {name: g[name] for name in OUT_CHANNELS}

        print(f"[PROCESS] {yyyymm}: days={len(days)} hours={len(out_times_idx)} engine={engine} layout={output_layout} grid=({grid_msg})")
        t_global = 0
        prev_accum_last: dict[str, np.ndarray] = {}

        for day_idx, (dd, tif_path) in enumerate(zip(days, tif_paths), start=1):
            day_t0 = time.perf_counter()
            print(f"  [DAY {day_idx:02d}/{len(days)}] {tif_path.name}")
            day_start = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:6]}-{dd}T00:00:00Z")
            hour_slice = slice(int((day_start - full_start) / pd.Timedelta(hours=1)), int((day_start - full_start) / pd.Timedelta(hours=1)) + 24)

            era5_day: dict[str, np.ndarray] = {}
            for name in ["t2m", "rh2m", "u10", "v10", "psfc", "tp_1h", "ssrd_1h", "strd_1h", *ERA5_DIRECT_VARS, "pblh"]:
                if name in era5_day:
                    continue
                vals = np.asarray(ds_era5[name].isel(time=hour_slice, latitude=era5_lat_sl, longitude=era5_lon_sl).values, dtype=np.float32)
                era5_day[name] = vals

            with rasterio.open(tif_path) as src:
                raw_stacks: dict[str, np.ndarray] = {}
                for land_var in LAND_BAND_VARS:
                    stack = np.full((24, len(land_lat), len(land_lon)), np.nan, dtype=np.float32)
                    for h in range(24):
                        band = src.read(land_band_index(land_var, h))[land_lat_sl, land_lon_sl]
                        stack[h] = mask_invalid(band)

                    if land_var in ACCUM_LAND_VARS:
                        prev_last = prev_accum_last.get(land_var)
                        if prev_last is None:
                            prev_last = load_prev_day_last_accum(f"{yyyymm}{dd}", land_var, land_lat_sl, land_lon_sl)
                        stack, prev_last_new = deaccumulate_stack(stack, prev_last)
                        prev_accum_last[land_var] = prev_last_new

                    raw_stacks[land_var] = stack

            land_vars = {
                "t2m": raw_stacks["temperature_2m"],
                "rh2m": dewpoint_to_rh(raw_stacks["temperature_2m"], raw_stacks["dewpoint_temperature_2m"]),
                "u10": raw_stacks["u_component_of_wind_10m"],
                "v10": raw_stacks["v_component_of_wind_10m"],
                "psfc": raw_stacks["surface_pressure"],
            }

            day_buf: dict[str, np.ndarray] = {}
            for name in ["t2m", "rh2m", "u10", "v10", "psfc"]:
                if output_layout == "full5179":
                    fb_on_target = interp_regular_grid(era5_day[name], era5_to_target_grid, device)
                    land_on_target = interp_regular_grid(land_vars[name], land_to_target_grid, device)
                    day_buf[name] = np.where(np.isfinite(land_on_target), land_on_target, fb_on_target).astype(np.float32)
                else:
                    fb_on_land = interp_regular_grid(era5_day[name], era5_to_land_grid, device)
                    day_buf[name] = np.where(np.isfinite(land_vars[name]), land_vars[name], fb_on_land).astype(np.float32)

            for name in ["tp_1h", "ssrd_1h", "strd_1h"]:
                if output_layout == "full5179":
                    day_buf[name] = interp_regular_grid(era5_day[name], era5_to_target_grid, device)
                else:
                    day_buf[name] = interp_regular_grid(era5_day[name], era5_to_land_grid, device)

            for name in ERA5_DIRECT_VARS:
                if output_layout == "full5179":
                    day_buf[name] = interp_regular_grid(era5_day[name], era5_to_target_grid, device)
                else:
                    day_buf[name] = interp_regular_grid(era5_day[name], era5_to_land_grid, device)

            day_buf["ws10"] = np.sqrt(day_buf["u10"] * day_buf["u10"] + day_buf["v10"] * day_buf["v10"]).astype(np.float32)
            day_buf["ventilation_index"] = (day_buf["pblh"] * day_buf["ws10"]).astype(np.float32)

            for name in OUT_CHANNELS:
                out_arrays[name][t_global : t_global + 24, :, :] = day_buf[name]

            t_global += 24
            day_elapsed = time.perf_counter() - day_t0
            print(f"    [DAY DONE] {yyyymm}{dd} elapsed={day_elapsed:.2f}s ({day_elapsed/60.0:.2f}m)")

        if ZARR_VERSION == 2:
            zarr.consolidate_metadata(str(out_path))
        month_elapsed = time.perf_counter() - month_t0
        print(f"[DONE] {yyyymm}: {out_path} elapsed={month_elapsed:.2f}s ({month_elapsed/60.0:.2f}m)")
        return out_path
    finally:
        ds_era5.close()


def main() -> None:
    args = parse_args()
    compressor = build_compressor(args.compressor, args.clevel, args.shuffle)
    print(f"[CONFIG] zarr_v{ZARR_VERSION} compressor={compressor_log_label(args.compressor, args.clevel, args.shuffle)}")
    start, end = parse_month_spec(args.month)
    out_root = Path(args.out_root)
    for yyyymm in iter_months(start, end):
        process_month(
            yyyymm,
            overwrite=args.overwrite,
            engine=args.engine,
            day_subset=args.days,
            out_root=out_root,
            compressor=compressor,
            output_layout=args.output_layout,
        )


if __name__ == "__main__":
    main()