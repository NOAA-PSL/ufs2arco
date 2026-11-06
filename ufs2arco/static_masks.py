"""Fit and validate immutable spatial masks from Anemoi training data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from ufs2arco.mpi import MPITopology, SerialTopology, _has_mpi


logger = logging.getLogger("ufs2arco")

_ARTIFACT_TYPE = "ufs2arco_training_static_mask"
_ARTIFACT_VERSION = 1
_METHODS = ("any_valid", "all_valid", "minimum_valid_fraction")
_MPI_WORLD_SIZE_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)


def _as_nonempty_unique_names(values: Sequence[str], caller: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{caller}: variables must be a sequence of strings")
    names = tuple(values)
    if len(names) == 0 or not all(
        isinstance(name, str) and name.strip() == name and name
        for name in names
    ):
        raise ValueError(
            f"{caller}: variables must contain non-empty strings without "
            "surrounding whitespace"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{caller}: variables must not contain duplicates")
    return names


def _validate_method(method: str, caller: str) -> str:
    if method not in _METHODS:
        raise ValueError(
            f"{caller}: method must be one of {_METHODS}; got {method!r}"
        )
    return method


def _validate_policy(
    method: str,
    minimum_valid_fraction: Optional[float],
    caller: str,
) -> tuple[str, Optional[float]]:
    method = _validate_method(method, caller)
    if method == "minimum_valid_fraction":
        if (
            isinstance(minimum_valid_fraction, (bool, np.bool_))
            or not isinstance(
                minimum_valid_fraction,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(minimum_valid_fraction)
            or not 0.0 < float(minimum_valid_fraction) <= 1.0
        ):
            raise ValueError(
                f"{caller}: minimum_valid_fraction must be finite and in "
                "(0, 1] when method is 'minimum_valid_fraction'"
            )
        return method, float(minimum_valid_fraction)

    if minimum_valid_fraction is not None:
        raise ValueError(
            f"{caller}: minimum_valid_fraction is only valid when method is "
            "'minimum_valid_fraction'"
        )
    return method, None


def _minimum_valid_count(
    minimum_valid_fraction: float,
    sample_count: int,
) -> int:
    """Return an exact ceiling for the user-facing decimal fraction."""

    product = Decimal(str(minimum_valid_fraction)) * sample_count
    return int(product.to_integral_value(rounding=ROUND_CEILING))


def _normalized_local_path(path: str | os.PathLike, caller: str) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"{caller}: path must be a string or os.PathLike")
    expanded = os.path.expanduser(os.path.expandvars(os.fspath(path)))
    if "://" in expanded:
        raise ValueError(
            f"{caller}: static-mask artifacts must use a local filesystem path"
        )
    return Path(expanded).resolve()


def _hash_array(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())


def grid_checksum(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    field_shape: Sequence[int],
    cells: Optional[np.ndarray] = None,
) -> str:
    """Return a checksum that is sensitive to grid values and cell ordering."""

    digest = hashlib.sha256()
    if cells is None:
        cells = np.arange(np.asarray(latitudes).size, dtype=np.int64)
    _hash_array(digest, np.asarray(cells))
    _hash_array(digest, np.asarray(latitudes))
    _hash_array(digest, np.asarray(longitudes))
    _hash_array(digest, np.asarray(field_shape, dtype=np.int64))
    return digest.hexdigest()


def _mask_checksum(
    valid: np.ndarray,
    variables: Sequence[str],
    method: str,
    minimum_valid_fraction: Optional[float] = None,
) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, np.asarray(valid, dtype=np.bool_))
    digest.update(json.dumps(list(variables)).encode("utf-8"))
    digest.update(method.encode("utf-8"))
    if minimum_valid_fraction is not None:
        digest.update(json.dumps(minimum_valid_fraction).encode("utf-8"))
    return digest.hexdigest()


def _json_checksum(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_identity(
    xds: xr.Dataset,
    source_path: Path,
    field_shape: Sequence[int],
) -> dict:
    """Return a small identity that every fitting rank must share."""

    dates = [str(pd.Timestamp(value)) for value in xds["dates"].values]
    missing_dates = sorted(
        str(pd.Timestamp(value))
        for value in xds.attrs.get("missing_dates", [])
    )
    identity = {
        "training_store": str(source_path),
        "data_dims": list(xds["data"].dims),
        "data_shape": list(xds["data"].shape),
        "data_dtype": str(xds["data"].dtype),
        "variables": list(xds.attrs.get("variables", [])),
        "field_shape": [int(value) for value in field_shape],
        "grid_checksum": grid_checksum(
            xds["latitudes"].values,
            xds["longitudes"].values,
            field_shape,
            xds["cell"].values,
        ),
        "dates_checksum": _json_checksum(dates),
        "missing_dates_checksum": _json_checksum(missing_dates),
        "latest_write_timestamp": xds.attrs.get("latest_write_timestamp"),
    }
    identity["identity_checksum"] = _json_checksum(identity)
    return identity


def _field_shape(xds: xr.Dataset, cell_count: int, caller: str) -> tuple[int, ...]:
    raw = xds.attrs.get("field_shape")
    if raw is None:
        raise ValueError(f"{caller}: dataset is missing attrs['field_shape']")
    shape = tuple(int(value) for value in raw)
    if len(shape) == 0 or np.prod(shape, dtype=np.int64) != cell_count:
        raise ValueError(
            f"{caller}: field_shape {shape} is incompatible with {cell_count} cells"
        )
    return shape


def _missing_time_indices(xds: xr.Dataset) -> np.ndarray:
    missing_dates = xds.attrs.get("missing_dates", [])
    if len(missing_dates) == 0:
        return np.empty(0, dtype=np.int64)

    dates = pd.DatetimeIndex(pd.to_datetime(xds["dates"].values))
    missing = pd.DatetimeIndex(pd.to_datetime(missing_dates))
    return np.flatnonzero(dates.isin(missing)).astype(np.int64)


def _validate_training_dataset(
    xds: xr.Dataset,
    variables: Sequence[str],
) -> tuple[list[int], tuple[int, ...]]:
    caller = "fit_static_mask_artifact"
    required = ("data", "dates", "latitudes", "longitudes")
    missing = [name for name in required if name not in xds]
    if missing:
        raise ValueError(
            f"{caller}: training dataset is missing required fields {missing}"
        )
    if "mean" not in xds:
        raise ValueError(
            f"{caller}: training dataset has not been finalized; aggregate "
            "statistics are missing"
        )

    data = xds["data"]
    required_dims = {"time", "variable", "cell"}
    if not required_dims.issubset(data.dims):
        raise ValueError(
            f"{caller}: data must contain dimensions {sorted(required_dims)}; "
            f"got {data.dims}"
        )
    unsupported_dims = set(data.dims) - {
        "time",
        "variable",
        "ensemble",
        "cell",
    }
    if unsupported_dims:
        raise ValueError(
            f"{caller}: data contains unsupported dimensions "
            f"{sorted(unsupported_dims)}; only time, variable, optional "
            "ensemble, and cell are supported"
        )
    if xds["dates"].dims != ("time",):
        raise ValueError(
            f"{caller}: dates must have exactly the 'time' dimension"
        )
    if xds["latitudes"].dims != ("cell",) or xds["longitudes"].dims != (
        "cell",
    ):
        raise ValueError(
            f"{caller}: latitudes and longitudes must both have dimension 'cell'"
        )

    available = list(xds.attrs.get("variables", []))
    if len(available) != data.sizes["variable"]:
        raise ValueError(
            f"{caller}: attrs['variables'] does not match the variable dimension"
        )
    absent = [name for name in variables if name not in available]
    if absent:
        raise KeyError(
            f"{caller}: variables {absent} are not in the training dataset; "
            f"available variables are {available}"
        )

    indices = [available.index(name) for name in variables]
    shape = _field_shape(xds, data.sizes["cell"], caller)
    return indices, shape


def _distributed_error(topo, local_error: Optional[str]) -> Optional[str]:
    gathered = topo.gather(local_error)
    if topo.is_root:
        combined = next(
            (error for error in gathered if error is not None),
            None,
        )
    else:
        combined = None
    return topo.broadcast(combined)


@dataclass(frozen=True)
class StaticMaskArtifact:
    """An immutable set of training-derived masks on one flattened grid."""

    path: str
    variables: tuple[str, ...]
    method: str
    valid: np.ndarray
    valid_count: np.ndarray
    valid_fraction: np.ndarray
    cells: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    field_shape: tuple[int, ...]
    sample_count: int
    training_start: str
    training_end: str
    training_identity_checksum: str
    grid_checksum: str
    mask_checksum: str
    minimum_valid_fraction: Optional[float] = None
    minimum_valid_count: Optional[int] = None

    def variable_index(self, variable: str) -> int:
        """Return an artifact variable index with a useful lookup error."""

        try:
            return self.variables.index(variable)
        except ValueError as exc:
            raise KeyError(
                f"Static mask artifact {self.path} does not contain {variable!r}; "
                f"available variables are {list(self.variables)}"
            ) from exc

    def valid_for(self, variable: str) -> np.ndarray:
        """Return a read-only Boolean validity vector for one variable."""

        values = self.valid[self.variable_index(variable)]
        values.setflags(write=False)
        return values

    def validate_grid(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        field_shape: Sequence[int],
        cells: Optional[np.ndarray] = None,
    ) -> None:
        """Reject any grid value, shape, or flattened-order mismatch."""

        actual_latitudes = np.asarray(latitudes)
        actual_longitudes = np.asarray(longitudes)
        if cells is None:
            cells = np.arange(actual_latitudes.size, dtype=np.int64)
        actual_cells = np.asarray(cells)
        actual_shape = tuple(int(value) for value in field_shape)
        if (
            actual_shape != self.field_shape
            or not np.array_equal(actual_cells, self.cells)
            or not np.array_equal(actual_latitudes, self.latitudes)
            or not np.array_equal(actual_longitudes, self.longitudes)
        ):
            actual_checksum = grid_checksum(
                actual_latitudes,
                actual_longitudes,
                actual_shape,
                actual_cells,
            )
            raise ValueError(
                "Static mask grid does not exactly match the target grid or "
                "flattened cell ordering: "
                f"artifact={self.grid_checksum}, target={actual_checksum}"
            )

    def manifest_entry(self, variable: str) -> dict:
        """Return JSON-safe provenance for an output dataset attribute."""

        self.variable_index(variable)
        entry = {
            "path": self.path,
            "variable": variable,
            "method": self.method,
            "sample_count": self.sample_count,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "training_identity_checksum": self.training_identity_checksum,
            "grid_checksum": self.grid_checksum,
            "mask_checksum": self.mask_checksum,
        }
        if self.minimum_valid_fraction is not None:
            entry["minimum_valid_fraction"] = self.minimum_valid_fraction
            entry["minimum_valid_count"] = self.minimum_valid_count
        return entry


def load_static_mask_artifact(
    path: str | os.PathLike,
) -> StaticMaskArtifact:
    """Load a complete static-mask artifact and verify its checksums."""

    caller = "load_static_mask_artifact"
    artifact_path = _normalized_local_path(path, caller)
    if not artifact_path.exists():
        raise FileNotFoundError(f"{caller}: artifact does not exist: {artifact_path}")

    # Validate the authoritative per-array and group metadata rather than
    # allowing a stale consolidated snapshot to hide a modified artifact.
    xds = xr.open_zarr(artifact_path, consolidated=False)
    try:
        attrs = xds.attrs
        if attrs.get("artifact_type") != _ARTIFACT_TYPE:
            raise ValueError(f"{caller}: {artifact_path} is not a UFS2ARCO mask artifact")
        if attrs.get("artifact_version") != _ARTIFACT_VERSION:
            raise ValueError(
                f"{caller}: unsupported artifact version "
                f"{attrs.get('artifact_version')!r}"
            )
        if attrs.get("complete") is not True:
            raise ValueError(f"{caller}: artifact is incomplete: {artifact_path}")

        required = (
            "valid",
            "valid_count",
            "valid_fraction",
            "latitudes",
            "longitudes",
        )
        missing = [name for name in required if name not in xds]
        if missing:
            raise ValueError(f"{caller}: artifact is missing fields {missing}")

        variables = _as_nonempty_unique_names(attrs.get("variables", []), caller)
        method, minimum_valid_fraction = _validate_policy(
            attrs.get("method"),
            attrs.get("minimum_valid_fraction"),
            caller,
        )
        loaded = xds[list(required)].load()
        valid = np.asarray(loaded["valid"].values, dtype=np.bool_)
        valid_count = np.asarray(loaded["valid_count"].values, dtype=np.int64)
        valid_fraction = np.asarray(
            loaded["valid_fraction"].values,
            dtype=np.float64,
        )
        latitudes = np.asarray(loaded["latitudes"].values)
        longitudes = np.asarray(loaded["longitudes"].values)
        cells = np.asarray(xds["cell"].values)
        field_shape = tuple(int(value) for value in attrs.get("field_shape", []))
        sample_count = int(attrs.get("sample_count", 0))
        minimum_valid_count = attrs.get("minimum_valid_count")
        training_identity = attrs.get("training_identity")
        if not isinstance(training_identity, dict):
            raise ValueError(f"{caller}: artifact training identity is missing")
        stored_identity_checksum = training_identity.get("identity_checksum")
        identity_payload = dict(training_identity)
        identity_payload.pop("identity_checksum", None)
        if stored_identity_checksum != _json_checksum(identity_payload):
            raise ValueError(f"{caller}: artifact training identity is invalid")

        expected_shape = (len(variables), latitudes.size)
        if valid.shape != expected_shape or valid_count.shape != expected_shape:
            raise ValueError(
                f"{caller}: artifact mask arrays have inconsistent shapes"
            )
        if valid_fraction.shape != expected_shape or sample_count <= 0:
            raise ValueError(f"{caller}: artifact counts are inconsistent")
        if minimum_valid_fraction is None:
            if minimum_valid_count is not None:
                raise ValueError(
                    f"{caller}: minimum_valid_count is only valid when method "
                    "is 'minimum_valid_fraction'"
                )
        else:
            expected_minimum_valid_count = _minimum_valid_count(
                minimum_valid_fraction,
                sample_count,
            )
            if (
                isinstance(minimum_valid_count, (bool, np.bool_))
                or not isinstance(minimum_valid_count, (int, np.integer))
                or int(minimum_valid_count) != expected_minimum_valid_count
            ):
                raise ValueError(
                    f"{caller}: artifact minimum_valid_count is inconsistent"
                )
            minimum_valid_count = int(minimum_valid_count)
        if longitudes.shape != latitudes.shape or cells.shape != latitudes.shape:
            raise ValueError(f"{caller}: artifact grid arrays have different shapes")
        if np.any(valid_count < 0) or np.any(valid_count > sample_count):
            raise ValueError(f"{caller}: artifact valid_count is out of range")
        expected_fraction = valid_count.astype(np.float64) / sample_count
        if not np.allclose(
            valid_fraction,
            expected_fraction,
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        ):
            raise ValueError(f"{caller}: artifact valid_fraction is inconsistent")
        if method == "any_valid":
            expected_valid = valid_count > 0
        elif method == "all_valid":
            expected_valid = valid_count == sample_count
        else:
            assert minimum_valid_count is not None
            expected_valid = valid_count >= minimum_valid_count
        if not np.array_equal(valid, expected_valid):
            raise ValueError(f"{caller}: artifact valid mask is inconsistent")

        actual_grid_checksum = grid_checksum(
            latitudes,
            longitudes,
            field_shape,
            cells,
        )
        stored_grid_checksum = attrs.get("grid_checksum")
        if actual_grid_checksum != stored_grid_checksum:
            raise ValueError(f"{caller}: artifact grid checksum is invalid")
        if training_identity.get("grid_checksum") != stored_grid_checksum:
            raise ValueError(
                f"{caller}: training identity and artifact grid differ"
            )
        if training_identity.get("field_shape") != list(field_shape):
            raise ValueError(
                f"{caller}: training identity and artifact field shape differ"
            )
        if training_identity.get("training_store") != attrs.get(
            "training_store"
        ):
            raise ValueError(
                f"{caller}: training identity and artifact source differ"
            )

        actual_mask_checksum = _mask_checksum(
            valid,
            variables,
            method,
            minimum_valid_fraction,
        )
        stored_mask_checksum = attrs.get("mask_checksum")
        if actual_mask_checksum != stored_mask_checksum:
            raise ValueError(f"{caller}: artifact mask checksum is invalid")

        for values in (
            valid,
            valid_count,
            valid_fraction,
            cells,
            latitudes,
            longitudes,
        ):
            values.setflags(write=False)

        return StaticMaskArtifact(
            path=str(artifact_path),
            variables=variables,
            method=method,
            valid=valid,
            valid_count=valid_count,
            valid_fraction=valid_fraction,
            cells=cells,
            latitudes=latitudes,
            longitudes=longitudes,
            field_shape=field_shape,
            sample_count=sample_count,
            training_start=str(attrs.get("training_start", "")),
            training_end=str(attrs.get("training_end", "")),
            training_identity_checksum=stored_identity_checksum,
            grid_checksum=stored_grid_checksum,
            mask_checksum=stored_mask_checksum,
            minimum_valid_fraction=minimum_valid_fraction,
            minimum_valid_count=minimum_valid_count,
        )
    finally:
        xds.close()


def _write_artifact(
    output_path: Path,
    artifact: xr.Dataset,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"A static-mask publisher already holds {lock_path}. If no fit "
            "process is running, remove that stale lock before retrying."
        ) from exc

    temporary_path = None
    try:
        with os.fdopen(lock_fd, "w") as lock_stream:
            json.dump({"pid": os.getpid(), "output": str(output_path)}, lock_stream)

        if output_path.exists():
            raise FileExistsError(
                f"Static mask artifacts are immutable and already exist at "
                f"{output_path}; choose a new versioned output path"
            )
        if overwrite:
            logger.warning(
                "overwrite has no effect for a new static-mask artifact; future "
                "replacement still requires a new versioned path"
            )

        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.name}.tmp-",
                dir=output_path.parent,
            )
        )
        artifact.to_zarr(temporary_path, mode="w")
        zarr.consolidate_metadata(temporary_path)
        # Validate the complete temporary store before atomically publishing it.
        # A malformed artifact must never occupy its immutable final path.
        load_static_mask_artifact(temporary_path)
        os.rename(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            shutil.rmtree(temporary_path)
        lock_path.unlink(missing_ok=True)


def fit_static_mask_artifact(
    training_store: str | os.PathLike,
    variables: Sequence[str],
    output_path: str | os.PathLike,
    method: str = "any_valid",
    overwrite: bool = False,
    topo=None,
    *,
    minimum_valid_fraction: Optional[float] = None,
) -> Optional[StaticMaskArtifact]:
    """Fit immutable per-variable masks from a finalized training Anemoi Zarr.

    ``any_valid`` is the inclusive policy: a cell is retained when the source
    variable is finite in at least one non-missing training sample.
    ``all_valid`` retains only cells finite in every such sample. Declared
    ``missing_dates`` are excluded from all reductions.
    ``minimum_valid_fraction`` retains cells whose finite sample count meets
    the ceiling of the configured fraction times the usable sample count.
    """

    caller = "fit_static_mask_artifact"
    names = _as_nonempty_unique_names(variables, caller)
    reduction, minimum_valid_fraction = _validate_policy(
        method,
        minimum_valid_fraction,
        caller,
    )
    source_path = _normalized_local_path(training_store, caller)
    destination = _normalized_local_path(output_path, caller)
    if (
        source_path == destination
        or source_path in destination.parents
        or destination in source_path.parents
    ):
        raise ValueError(
            f"{caller}: training store and artifact paths must not overlap"
        )
    if topo is None:
        topo = SerialTopology(log_dir=None)

    xds = None
    training_identity = None
    local_error = None
    try:
        xds = xr.open_zarr(source_path, consolidated=None)
        variable_indices, field_shape = _validate_training_dataset(xds, names)
        training_identity = _training_identity(
            xds,
            source_path,
            field_shape,
        )
        missing_indices = _missing_time_indices(xds)
        usable_indices = np.setdiff1d(
            np.arange(xds.sizes["time"], dtype=np.int64),
            missing_indices,
            assume_unique=True,
        )
        if usable_indices.size == 0:
            raise ValueError(f"{caller}: no usable training dates remain")

        rank_indices = np.array_split(usable_indices, topo.size)[topo.rank]
        local_count = np.zeros(
            (len(names), xds.sizes["cell"]),
            dtype=np.int64,
        )
        if rank_indices.size > 0:
            selected = xds["data"].isel(
                time=rank_indices,
                variable=variable_indices,
            )
            finite = np.isfinite(selected)
            reduction_dims = [
                dim
                for dim in finite.dims
                if dim not in ("variable", "cell")
            ]
            counts = finite.sum(reduction_dims).transpose("variable", "cell")
            # MPI already partitions time. A synchronous Dask scheduler keeps
            # every rank from spawning an additional node-wide thread pool.
            local_count[:] = counts.compute(scheduler="synchronous").values

        ensemble_count = xds.sizes.get("ensemble", 1)
        local_sample_count = np.asarray(
            [rank_indices.size * ensemble_count],
            dtype=np.int64,
        )
    except Exception as exc:
        local_error = f"rank {topo.rank}: {type(exc).__name__}: {exc}"

    error = _distributed_error(topo, local_error)
    if error is not None:
        if xds is not None:
            xds.close()
        raise RuntimeError(f"{caller}: distributed fit failed: {error}")

    gathered_identities = topo.gather(training_identity)
    identity_error = None
    if topo.is_root:
        reference_identity = gathered_identities[0]
        mismatch = next(
            (
                rank
                for rank, identity in enumerate(gathered_identities[1:], start=1)
                if identity != reference_identity
            ),
            None,
        )
        if mismatch is not None:
            identity_error = (
                "training-store identity differs between rank 0 and rank "
                f"{mismatch}; stop all writers and fit from one finalized store"
            )

    identity_error = topo.broadcast(identity_error)
    if identity_error is not None:
        if xds is not None:
            xds.close()
        raise RuntimeError(f"{caller}: distributed fit failed: {identity_error}")

    global_count = np.zeros_like(local_count)
    global_sample_count = np.zeros(1, dtype=np.int64)
    topo.sum(local_count, global_count)
    topo.sum(local_sample_count, global_sample_count)

    artifact_result = None
    write_error = None
    if topo.is_root:
        try:
            sample_count = int(global_sample_count[0])
            if sample_count <= 0:
                raise ValueError(f"{caller}: no training samples were counted")

            minimum_valid_count = None
            if reduction == "any_valid":
                valid = global_count > 0
            elif reduction == "all_valid":
                valid = global_count == sample_count
            else:
                assert minimum_valid_fraction is not None
                minimum_valid_count = _minimum_valid_count(
                    minimum_valid_fraction,
                    sample_count,
                )
                valid = global_count >= minimum_valid_count
            valid_fraction = global_count.astype(np.float64) / sample_count

            latitudes = np.asarray(xds["latitudes"].values)
            longitudes = np.asarray(xds["longitudes"].values)
            cells = np.asarray(xds["cell"].values)
            checksum_grid = grid_checksum(
                latitudes,
                longitudes,
                field_shape,
                cells,
            )
            checksum_mask = _mask_checksum(
                valid,
                names,
                reduction,
                minimum_valid_fraction,
            )
            dates = pd.DatetimeIndex(pd.to_datetime(xds["dates"].values))
            used_dates = dates[usable_indices]

            artifact_attrs = {
                "artifact_type": _ARTIFACT_TYPE,
                "artifact_version": _ARTIFACT_VERSION,
                "complete": True,
                "variables": list(names),
                "method": reduction,
                "validity_rule": "isfinite",
                "reduction_dimensions": ["time", "ensemble"],
                "missing_dates_excluded": int(missing_indices.size),
                "sample_count": sample_count,
                "training_store": str(source_path),
                "training_identity": training_identity,
                "training_start": str(used_dates[0]),
                "training_end": str(used_dates[-1]),
                "field_shape": list(field_shape),
                "grid_checksum": checksum_grid,
                "mask_checksum": checksum_mask,
            }
            if minimum_valid_fraction is not None:
                artifact_attrs["minimum_valid_fraction"] = minimum_valid_fraction
                artifact_attrs["minimum_valid_count"] = minimum_valid_count

            artifact = xr.Dataset(
                {
                    "valid": (
                        ("mask_variable", "cell"),
                        valid.astype(np.bool_),
                    ),
                    "valid_count": (
                        ("mask_variable", "cell"),
                        global_count,
                    ),
                    "valid_fraction": (
                        ("mask_variable", "cell"),
                        valid_fraction,
                    ),
                    "latitudes": ("cell", latitudes),
                    "longitudes": ("cell", longitudes),
                },
                coords={
                    "mask_variable": np.arange(len(names), dtype=np.int64),
                    "cell": np.asarray(xds["cell"].values),
                },
                attrs=artifact_attrs,
            )
            _write_artifact(destination, artifact, overwrite=overwrite)
            artifact_result = load_static_mask_artifact(destination)

            for index, name in enumerate(names):
                counts = global_count[index]
                never = int(np.count_nonzero(counts == 0))
                always = int(np.count_nonzero(counts == sample_count))
                intermittent = int(counts.size - never - always)
                retained = int(np.count_nonzero(valid[index]))
                logger.info(
                    "%s: %s retained=%d always_valid=%d intermittent=%d "
                    "never_valid=%d",
                    caller,
                    name,
                    retained,
                    always,
                    intermittent,
                    never,
                )
        except Exception as exc:
            write_error = f"{type(exc).__name__}: {exc}"

    write_error = topo.broadcast(write_error)
    if xds is not None:
        xds.close()
    if write_error is not None:
        raise RuntimeError(f"{caller}: artifact write failed: {write_error}")
    topo.barrier()
    return artifact_result


def _command_topology(log_dir: Optional[str]):
    launcher_size = max(
        (
            int(os.environ.get(name, "1"))
            for name in _MPI_WORLD_SIZE_ENV
            if os.environ.get(name, "1").isdigit()
        ),
        default=1,
    )
    if _has_mpi:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            return MPITopology(log_dir=log_dir)
    if launcher_size > 1:
        raise RuntimeError(
            "Static-mask fitting was launched with multiple processes, but "
            "mpi4py did not provide a matching MPI communicator. Install the "
            "ufs2arco MPI extra and use the cluster MPI build; refusing to let "
            "independent serial processes race on one artifact."
        )
    return SerialTopology(log_dir=log_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point for fitting a static-mask artifact."""

    parser = argparse.ArgumentParser(
        description=(
            "Fit immutable spatial masks from a finalized Anemoi training Zarr"
        )
    )
    parser.add_argument("training_store")
    parser.add_argument("output_path")
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--method", choices=_METHODS, default="any_valid")
    parser.add_argument("--minimum-valid-fraction", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)

    topo = _command_topology(args.log_dir)
    artifact = fit_static_mask_artifact(
        training_store=args.training_store,
        variables=args.variables,
        output_path=args.output_path,
        method=args.method,
        minimum_valid_fraction=args.minimum_valid_fraction,
        overwrite=args.overwrite,
        topo=topo,
    )
    if topo.is_root:
        print(
            f"Static mask artifact ready at {artifact.path} "
            f"(checksum={artifact.mask_checksum})"
        )


if __name__ == "__main__":
    main()
