#!/usr/bin/env python
"""Diagnose missing dates, NaNs, and missing chunks in an Anemoi zarr dataset.

This script is meant to run where the zarr store is mounted. It uses the
metadata arrays written by the ufs2arco Anemoi target when available, so the
default checks avoid reading the full data cube.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Iterable

np = None
pd = None
xr = None


def import_runtime_dependencies() -> None:
    """Import heavy dependencies after argparse so --help works in bare Python."""
    global np, pd, xr
    import numpy as numpy_module
    import pandas as pandas_module
    import xarray as xarray_module

    np = numpy_module
    pd = pandas_module
    xr = xarray_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report missing dates, variables/dates with any NaNs, variables/dates "
            "that are fully NaN, and optionally missing zarr chunk files."
        )
    )
    parser.add_argument("zarr", help="Path to the Anemoi zarr store")
    parser.add_argument(
        "--data-var",
        default="data",
        help="Name of the main data variable in the zarr store. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV outputs. Default: ./anemoi_diagnostics_<zarr-name>",
    )
    parser.add_argument(
        "--freq",
        default=None,
        help=(
            "Expected time frequency for detecting missing coordinate dates, e.g. "
            "'6h'. Default: infer from the median date spacing."
        ),
    )
    parser.add_argument(
        "--raw-scan",
        action="store_true",
        help=(
            "Scan the raw data variable if count_array/has_nans_array are missing. "
            "This can be expensive for large datasets."
        ),
    )
    parser.add_argument(
        "--check-chunks",
        action="store_true",
        help=(
            "For local filesystem zarr stores, inspect data chunk files and report "
            "missing chunk groups by date/variable chunk range."
        ),
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=25,
        help="Maximum rows to print for each issue category. CSVs include all rows.",
    )
    return parser.parse_args()


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def stringify_datetime(value) -> str:
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def get_dates(ds: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    if "dates" in ds:
        raw_dates = ds["dates"].values
    elif "date" in ds.coords:
        raw_dates = ds["date"].values
    elif "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        raw_dates = ds["time"].values
    elif "time" in ds.dims:
        raw_dates = np.arange(ds.sizes["time"])
    else:
        raise ValueError("Could not find a dates variable, date coord, or time dimension")

    return raw_dates, [stringify_datetime(value) for value in raw_dates]


def get_variable_names(ds: xr.Dataset) -> list[str]:
    variables = as_list(ds.attrs.get("variables"))
    if variables:
        return [str(value) for value in variables]

    if "variable" in ds.coords:
        return [str(value) for value in ds["variable"].values]

    if "variable" in ds.dims:
        return [f"variable_{idx}" for idx in range(ds.sizes["variable"])]

    raise ValueError("Could not find variable names in attrs or a variable dimension")


def normalize_flag_array(
    arr: xr.DataArray,
    *,
    date_strings: list[str],
    variable_names: list[str],
) -> xr.DataArray:
    """Attach readable date/variable coords without changing dimension names."""
    if "time" in arr.dims:
        arr = arr.assign_coords(date_label=("time", date_strings))
    if "dates" in arr.dims:
        arr = arr.assign_coords(date_label=("dates", date_strings))
    if "variable" in arr.dims:
        arr = arr.assign_coords(variable_name=("variable", variable_names))
    return arr


def horizontal_dims(data: xr.DataArray) -> list[str]:
    non_horizontal = {"time", "dates", "date", "variable", "ensemble"}
    return [dim for dim in data.dims if dim not in non_horizontal]


def get_nan_flags(
    ds: xr.Dataset,
    data_var: str,
    *,
    raw_scan: bool,
) -> tuple[xr.DataArray | None, xr.DataArray | None, str]:
    """Return (has_any_nan, fully_nan, source_description)."""
    if "has_nans_array" in ds:
        has_any_nan = ds["has_nans_array"].astype(bool)
    elif raw_scan:
        data = ds[data_var]
        has_any_nan = np.isnan(data).any(horizontal_dims(data))
    else:
        has_any_nan = None

    if "count_array" in ds:
        fully_nan = ds["count_array"] == 0
    elif raw_scan:
        data = ds[data_var]
        fully_nan = np.isnan(data).all(horizontal_dims(data))
    else:
        fully_nan = None

    if "has_nans_array" in ds or "count_array" in ds:
        source = "ufs2arco metadata arrays"
    elif raw_scan:
        source = f"raw {data_var!r} scan"
    else:
        source = "not available; rerun with --raw-scan"
    return has_any_nan, fully_nan, source


def flag_rows(
    flag: xr.DataArray,
    *,
    date_strings: list[str],
    variable_names: list[str],
) -> pd.DataFrame:
    """Convert True entries in a flag array to a tidy dataframe."""
    flag = normalize_flag_array(
        flag,
        date_strings=date_strings,
        variable_names=variable_names,
    )
    values = flag.compute().values.astype(bool)
    if values.size == 0 or not values.any():
        return pd.DataFrame()

    rows = []
    true_indices = np.argwhere(values)
    dims = list(flag.dims)
    coords = {dim: flag[dim].values for dim in dims if dim in flag.coords}

    for index_tuple in true_indices:
        row = {}
        for dim, idx in zip(dims, index_tuple):
            if dim in ("time", "dates", "date"):
                row["time_index"] = int(idx)
                row["date"] = date_strings[int(idx)]
            elif dim == "variable":
                row["variable_index"] = int(idx)
                row["variable"] = variable_names[int(idx)]
            elif dim == "ensemble":
                value = coords.get(dim, np.arange(flag.sizes[dim]))[int(idx)]
                row["ensemble"] = str(value)
            else:
                value = coords.get(dim, np.arange(flag.sizes[dim]))[int(idx)]
                row[dim] = str(value)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_date_variable(flag: xr.DataArray, mode: str) -> xr.DataArray:
    """Aggregate a flag array to date x variable when it has extra dims."""
    extra_dims = [dim for dim in flag.dims if dim not in ("time", "dates", "date", "variable")]
    if not extra_dims:
        return flag
    if mode == "any":
        return flag.any(extra_dims)
    if mode == "all":
        return flag.all(extra_dims)
    raise ValueError(f"Unknown aggregation mode: {mode}")


def summarize_flags(
    has_any_nan: xr.DataArray | None,
    fully_nan: xr.DataArray | None,
    *,
    date_strings: list[str],
    variable_names: list[str],
    output_dir: Path,
    max_print: int,
) -> None:
    if has_any_nan is not None:
        has_any_nan = normalize_flag_array(
            has_any_nan,
            date_strings=date_strings,
            variable_names=variable_names,
        )
        rows = flag_rows(
            aggregate_date_variable(has_any_nan, "any"),
            date_strings=date_strings,
            variable_names=variable_names,
        )
        write_and_preview(
            rows,
            output_dir / "date_variable_any_nan.csv",
            "date/variable entries with at least one NaN cell",
            max_print,
        )

        if not rows.empty:
            var_summary = rows.groupby("variable").size().reset_index(name="date_count")
            var_summary = var_summary.sort_values(["date_count", "variable"], ascending=[False, True])
            write_and_preview(
                var_summary,
                output_dir / "variables_with_any_nan.csv",
                "variables with any NaN cells",
                max_print,
            )

            date_summary = rows.groupby("date").size().reset_index(name="variable_count")
            date_summary = date_summary.sort_values(["variable_count", "date"], ascending=[False, True])
            write_and_preview(
                date_summary,
                output_dir / "dates_with_any_nan.csv",
                "dates with any NaN cells",
                max_print,
            )

    if fully_nan is not None:
        fully_nan = normalize_flag_array(
            fully_nan,
            date_strings=date_strings,
            variable_names=variable_names,
        )
        per_variable_all_nan = fully_nan.all([dim for dim in fully_nan.dims if dim != "variable"])
        per_variable_all_nan_rows = flag_rows(
            per_variable_all_nan,
            date_strings=date_strings,
            variable_names=variable_names,
        )
        write_and_preview(
            per_variable_all_nan_rows,
            output_dir / "variables_fully_nan_entire_dataset.csv",
            "variables fully NaN for every date/ensemble/cell",
            max_print,
        )

        per_ensemble_rows = flag_rows(
            fully_nan,
            date_strings=date_strings,
            variable_names=variable_names,
        )
        write_and_preview(
            per_ensemble_rows,
            output_dir / "date_variable_ensemble_fully_nan.csv",
            "date/variable/ensemble entries that are fully NaN",
            max_print,
        )

        date_variable_all_ensembles = aggregate_date_variable(fully_nan, "all")
        rows = flag_rows(
            date_variable_all_ensembles,
            date_strings=date_strings,
            variable_names=variable_names,
        )
        write_and_preview(
            rows,
            output_dir / "date_variable_fully_nan.csv",
            "date/variable entries fully NaN across all ensembles/cells",
            max_print,
        )

        if not rows.empty:
            var_summary = rows.groupby("variable").size().reset_index(name="date_count")
            var_summary = var_summary.sort_values(["date_count", "variable"], ascending=[False, True])
            write_and_preview(
                var_summary,
                output_dir / "variables_with_fully_nan_dates.csv",
                "variables with fully NaN dates",
                max_print,
            )

            date_summary = rows.groupby("date").size().reset_index(name="variable_count")
            date_summary = date_summary.sort_values(["variable_count", "date"], ascending=[False, True])
            write_and_preview(
                date_summary,
                output_dir / "dates_with_fully_nan_variables.csv",
                "dates with fully NaN variables",
                max_print,
            )

            fully_missing_dates = date_summary[
                date_summary["variable_count"] == len(variable_names)
            ].copy()
            write_and_preview(
                fully_missing_dates,
                output_dir / "dates_all_variables_fully_nan.csv",
                "dates where all variables are fully NaN",
                max_print,
            )


def write_and_preview(df: pd.DataFrame, path: Path, title: str, max_print: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n{title}: {len(df)}")
    print(f"  wrote {path}")
    if df.empty:
        return
    preview = df.head(max_print)
    print(preview.to_string(index=False))
    if len(df) > max_print:
        print(f"  ... {len(df) - max_print} more rows in CSV")


def date_gap_report(
    raw_dates: np.ndarray,
    date_strings: list[str],
    *,
    freq: str | None,
) -> pd.DataFrame:
    raw_dates_array = np.asarray(raw_dates)
    if np.issubdtype(raw_dates_array.dtype, np.number):
        return pd.DataFrame()

    try:
        dates = pd.to_datetime(raw_dates)
    except Exception:
        return pd.DataFrame()

    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) < 2:
        return pd.DataFrame()

    if freq is None:
        diffs = dates.to_series().diff().dropna()
        freq = diffs.median()

    expected = pd.date_range(dates.min(), dates.max(), freq=freq)
    missing = expected.difference(dates)
    return pd.DataFrame({"date": [value.isoformat() for value in missing]})


def print_missing_dates_attr(ds: xr.Dataset, output_dir: Path, max_print: int) -> None:
    missing_dates = [stringify_datetime(value) for value in as_list(ds.attrs.get("missing_dates"))]
    df = pd.DataFrame({"date": missing_dates})
    write_and_preview(df, output_dir / "attr_missing_dates.csv", "missing_dates stored in zarr attrs", max_print)


def chunk_counts(shape: Iterable[int], chunks: Iterable[int]) -> list[int]:
    return [math.ceil(size / chunk) for size, chunk in zip(shape, chunks)]


def chunk_label(
    *,
    axis_name: str,
    chunk_index: int,
    chunk_size: int,
    axis_size: int,
    date_strings: list[str],
    variable_names: list[str],
) -> str:
    start = chunk_index * chunk_size
    stop = min(start + chunk_size, axis_size)
    if axis_name == "time":
        if stop == start + 1:
            return date_strings[start]
        return f"{date_strings[start]} .. {date_strings[stop - 1]}"
    if axis_name == "variable":
        if stop == start + 1:
            return variable_names[start]
        return f"{variable_names[start]} .. {variable_names[stop - 1]}"
    return f"{axis_name}[{start}:{stop})"


def iter_local_chunk_indices(array_dir: Path, ndim: int, dimension_separator: str) -> Iterable[tuple[int, ...]]:
    for root, _, files in os.walk(array_dir):
        root_path = Path(root)
        for name in files:
            if name.startswith("."):
                continue
            rel = (root_path / name).relative_to(array_dir)
            if dimension_separator == "/":
                parts = rel.parts
            else:
                parts = str(rel).split(dimension_separator)
            if len(parts) != ndim:
                continue
            try:
                yield tuple(int(part) for part in parts)
            except ValueError:
                continue


def check_missing_chunks(
    zarr_path: Path,
    data_var: str,
    ds: xr.Dataset,
    *,
    date_strings: list[str],
    variable_names: list[str],
    output_dir: Path,
    max_print: int,
) -> None:
    array_dir = zarr_path / data_var
    metadata_path = array_dir / ".zarray"
    if not metadata_path.exists():
        print(f"\nmissing chunk check: skipped; {metadata_path} was not found")
        return

    with metadata_path.open() as f:
        metadata = json.load(f)

    shape = list(metadata["shape"])
    chunks = list(metadata["chunks"])
    dimension_separator = metadata.get("dimension_separator", ".")
    nchunks = chunk_counts(shape, chunks)

    data_dims = list(ds[data_var].dims)
    leading_dims = [dim for dim in ("time", "variable", "ensemble") if dim in data_dims]
    leading_axes = [data_dims.index(dim) for dim in leading_dims]
    tail_axes = [axis for axis in range(len(shape)) if axis not in leading_axes]
    expected_tail_chunks = math.prod(nchunks[axis] for axis in tail_axes)

    present_by_prefix: dict[tuple[int, ...], int] = {}
    for chunk_index in iter_local_chunk_indices(array_dir, len(shape), dimension_separator):
        prefix = tuple(chunk_index[axis] for axis in leading_axes)
        present_by_prefix[prefix] = present_by_prefix.get(prefix, 0) + 1

    rows = []
    leading_ranges = [range(nchunks[axis]) for axis in leading_axes]
    for prefix in itertools.product(*leading_ranges):
        present = present_by_prefix.get(prefix, 0)
        missing = expected_tail_chunks - present
        if missing <= 0:
            continue

        row = {
            "present_chunks": present,
            "expected_chunks": expected_tail_chunks,
            "missing_chunks": missing,
        }
        for dim, axis, chunk_index in zip(leading_dims, leading_axes, prefix):
            row[f"{dim}_chunk"] = chunk_index
            row[dim] = chunk_label(
                axis_name=dim,
                chunk_index=chunk_index,
                chunk_size=chunks[axis],
                axis_size=shape[axis],
                date_strings=date_strings,
                variable_names=variable_names,
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_columns = [column for column in ("missing_chunks", "time_chunk", "variable_chunk") if column in df]
        df = df.sort_values(sort_columns, ascending=[False] + [True] * (len(sort_columns) - 1))
    write_and_preview(
        df,
        output_dir / "missing_data_chunk_prefixes.csv",
        "date/variable chunk groups with missing data chunk files",
        max_print,
    )


def main() -> None:
    args = parse_args()
    import_runtime_dependencies()

    zarr_path = Path(args.zarr)
    output_dir = Path(args.output_dir or f"anemoi_diagnostics_{zarr_path.name.replace('.zarr', '')}")

    ds = xr.open_zarr(args.zarr)
    if args.data_var not in ds:
        raise KeyError(f"{args.data_var!r} was not found in {args.zarr}")

    raw_dates, date_strings = get_dates(ds)
    variable_names = get_variable_names(ds)

    print(f"dataset: {args.zarr}")
    print(f"output:  {output_dir}")
    print(f"dims:    {dict(ds.sizes)}")
    print(f"vars:    {len(variable_names)}")
    print(f"dates:   {len(date_strings)} ({date_strings[0]} .. {date_strings[-1]})")
    print(f"chunks:  {getattr(ds[args.data_var].data, 'chunks', None)}")

    print_missing_dates_attr(ds, output_dir, args.max_print)

    gaps = date_gap_report(raw_dates, date_strings, freq=args.freq)
    write_and_preview(
        gaps,
        output_dir / "coordinate_date_gaps.csv",
        "missing dates in the date coordinate itself",
        args.max_print,
    )

    has_any_nan, fully_nan, source = get_nan_flags(ds, args.data_var, raw_scan=args.raw_scan)
    print(f"\nNaN flag source: {source}")
    summarize_flags(
        has_any_nan,
        fully_nan,
        date_strings=date_strings,
        variable_names=variable_names,
        output_dir=output_dir,
        max_print=args.max_print,
    )

    if args.check_chunks:
        check_missing_chunks(
            zarr_path,
            args.data_var,
            ds,
            date_strings=date_strings,
            variable_names=variable_names,
            output_dir=output_dir,
            max_print=args.max_print,
        )


if __name__ == "__main__":
    main()
