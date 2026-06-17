from __future__ import annotations

import argparse
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr


UTC = timezone.utc
MONTH_RE = re.compile(r"^(\d{6})(?:-(\d{6}))?$")

ROOT = Path(__file__).resolve().parents[1]
IN_ROOT_DEFAULT = ROOT / "2003_airkorea"
OUT_ROOT_DEFAULT = ROOT / "2007_airkorea_parquet"
POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build 2007 monthly wide parquet from 2003 AirKorea zarr")
    ap.add_argument("--month", required=True, help="YYYYMM or YYYYMM-YYYYMM")
    ap.add_argument("--in-root", type=Path, default=IN_ROOT_DEFAULT, help="Input 2003 root")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT, help="Output 2007 parquet root")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing month partition")
    ap.add_argument("--files-per-month", type=int, default=1, help="Output parquet files per month (1-4 recommended)")
    ap.add_argument("--row-group-size", type=int, default=131072, help="Parquet row group size")
    ap.add_argument("--compression", choices=["zstd", "snappy", "gzip", "none"], default="zstd", help="Parquet compression")
    ap.add_argument("--compression-level", type=int, default=3, help="Compression level (codec-dependent)")
    ap.add_argument("--time-chunk", type=int, default=168, help="Time chunk for zarr reads")
    return ap.parse_args()


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


def month_mask(ts_utc: pd.DatetimeIndex, yyyymm: str) -> np.ndarray:
    return (ts_utc.year == int(yyyymm[:4])) & (ts_utc.month == int(yyyymm[4:6]))


def load_inputs(in_root: Path):
    if not in_root.exists():
        raise RuntimeError(f"missing input root: {in_root}")

    station_path = in_root / "station_grid.parquet"
    time_path = in_root / "time_index.parquet"
    if not station_path.exists() or not time_path.exists():
        raise RuntimeError(f"missing station/time parquet in {in_root}")

    station_df = pd.read_parquet(station_path)
    time_df = pd.read_parquet(time_path)
    if "ts" not in time_df.columns:
        raise RuntimeError(f"time_index.parquet missing ts column: {time_path}")
    for col in ["station_id", "i", "j"]:
        if col not in station_df.columns:
            raise RuntimeError(f"station_grid.parquet missing {col}: {station_path}")

    ts_utc = pd.DatetimeIndex(pd.to_datetime(time_df["ts"], utc=True, errors="coerce"))
    if ts_utc.isna().any():
        raise RuntimeError("time_index has invalid timestamps")

    stores: dict[str, zarr.Array] = {}
    n_t_expected = len(ts_utc)
    for poll in POLLUTANTS:
        p = in_root / f"{poll}.zarr"
        if not p.exists():
            raise RuntimeError(f"missing zarr: {p}")
        z = zarr.open(str(p), mode="r")
        if len(z.shape) != 3:
            raise RuntimeError(f"unexpected shape for {p.name}: {z.shape}; expected (time,y,x)")
        if int(z.shape[0]) != n_t_expected:
            raise RuntimeError(f"time mismatch for {p.name}: {z.shape[0]} vs {n_t_expected}")
        stores[poll] = z

    station_df = station_df.copy()
    station_df["station_id"] = station_df["station_id"].astype(str)
    station_df["i"] = station_df["i"].astype(np.int32)
    station_df["j"] = station_df["j"].astype(np.int32)
    return station_df, ts_utc, stores


def extract_poll_month_values(
    z: zarr.Array,
    month_indices: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    time_chunk: int,
) -> np.ndarray:
    n_station = int(i_idx.size)
    out_parts: list[np.ndarray] = []

    if month_indices.size == 0:
        return np.empty((0,), dtype=np.float32)

    if np.any(np.diff(month_indices) != 1):
        raise RuntimeError("month time index is not contiguous")

    start = int(month_indices[0])
    end = int(month_indices[-1]) + 1

    for t0 in range(start, end, max(1, int(time_chunk))):
        t1 = min(t0 + max(1, int(time_chunk)), end)
        cube = np.asarray(z[t0:t1, :, :], dtype=np.float32)
        sampled = cube[:, i_idx, j_idx]
        out_parts.append(sampled.reshape(-1))

    if not out_parts:
        return np.empty((0,), dtype=np.float32)

    vals = np.concatenate(out_parts, axis=0).astype(np.float32, copy=False)
    if vals.size != (month_indices.size * n_station):
        raise RuntimeError("unexpected sampled size")
    return vals


def build_month_dataframe(
    yyyymm: str,
    ts_utc: pd.DatetimeIndex,
    station_df: pd.DataFrame,
    stores: dict[str, zarr.Array],
    time_chunk: int,
) -> pd.DataFrame:
    m_mask = month_mask(ts_utc, yyyymm)
    month_idx = np.where(m_mask)[0].astype(np.int64)
    if month_idx.size == 0:
        raise RuntimeError(f"no timestamps for month={yyyymm}")

    ts_month = ts_utc[month_idx]
    n_t = int(month_idx.size)
    station_ids = station_df["station_id"].to_numpy(dtype=object)
    i_idx = station_df["i"].to_numpy(dtype=np.int32)
    j_idx = station_df["j"].to_numpy(dtype=np.int32)
    n_s = int(station_ids.size)
    n_rows = n_t * n_s

    ts_rep = np.repeat(np.asarray(ts_month.values), n_s)
    station_rep = np.tile(station_ids, n_t)
    gy_rep = np.tile(i_idx, n_t)
    gx_rep = np.tile(j_idx, n_t)

    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(ts_rep, utc=True),
            "station_id": station_rep,
            "grid_y": gy_rep.astype(np.int32),
            "grid_x": gx_rep.astype(np.int32),
            "month": np.full(n_rows, yyyymm, dtype=object),
        }
    )

    for poll in POLLUTANTS:
        vals = extract_poll_month_values(stores[poll], month_idx, i_idx, j_idx, time_chunk=time_chunk)
        valid = np.isfinite(vals)
        out[poll] = vals.astype(np.float32)
        out[f"valid_{poll}"] = valid.astype(np.uint8)

    out = out.sort_values(["timestamp_utc", "station_id"], kind="mergesort", ignore_index=True)
    return out


def write_month_partition(
    df: pd.DataFrame,
    yyyymm: str,
    out_root: Path,
    overwrite: bool,
    files_per_month: int,
    row_group_size: int,
    compression: str,
    compression_level: int,
) -> None:
    part_dir = out_root / yyyymm
    if part_dir.exists():
        if not overwrite:
            print(f"[SKIP] {yyyymm}: exists {part_dir}")
            return
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    n_files = max(1, min(4, int(files_per_month)))
    n_rows = len(df)
    rows_per_file = int(math.ceil(n_rows / n_files))

    codec = None if compression == "none" else compression
    for fi in range(n_files):
        s = fi * rows_per_file
        e = min((fi + 1) * rows_per_file, n_rows)
        if s >= e:
            break
        sub = df.iloc[s:e].reset_index(drop=True)
        table = pa.Table.from_pandas(sub, preserve_index=False)
        out_path = part_dir / f"part-{fi:03d}.parquet"
        pq.write_table(
            table,
            out_path,
            compression=codec,
            compression_level=int(compression_level) if codec is not None else None,
            row_group_size=max(1, int(row_group_size)),
            use_dictionary=["station_id", "month"],
            write_statistics=True,
        )


def main() -> None:
    args = parse_args()
    start, end = parse_month_spec(args.month)

    station_df, ts_utc, stores = load_inputs(args.in_root)
    print(f"[CONFIG] in={args.in_root} out={args.out_root}")
    print(
        f"[CONFIG] files_per_month={max(1, min(4, int(args.files_per_month)))} "
        f"row_group_size={max(1, int(args.row_group_size))} "
        f"compression={args.compression} clevel={int(args.compression_level)}"
    )

    for yyyymm in iter_months(start, end):
        print(f"[PROCESS] {yyyymm} ...")
        df = build_month_dataframe(
            yyyymm=yyyymm,
            ts_utc=ts_utc,
            station_df=station_df,
            stores=stores,
            time_chunk=max(1, int(args.time_chunk)),
        )
        write_month_partition(
            df=df,
            yyyymm=yyyymm,
            out_root=args.out_root,
            overwrite=args.overwrite,
            files_per_month=args.files_per_month,
            row_group_size=args.row_group_size,
            compression=args.compression,
            compression_level=args.compression_level,
        )
        valid_cols = [f"valid_{p}" for p in POLLUTANTS]
        valid_total = int(df[valid_cols].to_numpy(dtype=np.uint8).sum())
        print(
            f"[DONE] {yyyymm}: rows={len(df):,} "
            f"valid_total={valid_total:,} "
            f"unique_ts={df['timestamp_utc'].nunique()} stations={df['station_id'].nunique()}"
        )


if __name__ == "__main__":
    main()