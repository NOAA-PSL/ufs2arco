"""Ocean diagnostics computed from temperature and salinity profiles.

Pure numpy implementations of seawater density, mixed layer depth, and ocean heat
content. There is no xarray or dask in this module; see
:mod:`ufs2arco.transforms.ocean` for the dataset-level wrappers.

The algorithms are ported from GFDL MOM6, at commit
``9f50395665006015dc966cace7a42800f2229af3`` of https://github.com/NOAA-GFDL/MOM6.
Each function names the MOM6 source file it came from, and any deliberate
deviation is called out in that function's docstring.

The module is named for the science rather than the model, so that a TEOS-10
equation of state, an energy based mixed layer depth, or a non-MOM6 ocean source
can be added here later without the filename becoming a lie.

All routines that reduce over the vertical expect arrays shaped ``(..., nz)``,
which is the layout ``xarray.apply_ufunc`` produces for a single input core
dimension.
"""

import logging

import numpy as np

logger = logging.getLogger("ufs2arco")

# Wright (1997) equation of state coefficients, transcribed verbatim from
# MOM6 src/equation_of_state/MOM_EOS_Wright_full.F90 and MOM_EOS_Wright_red.F90.
#
# Note that MOM6 also ships MOM_EOS_Wright.F90, whose own module docstring
# describes it as "a poor implementation (missing parenthesis and bugs) of the
# reduced range Wright 1997 expressions" and whose type is named
# buggy_Wright_EOS. It is kept upstream only to reproduce old results, and is
# deliberately not implemented here.
#
# Order is (a0, a1, a2, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5).
_WRIGHT_COEFFICIENTS = {
    # Fit to UNESCO over the full range: -2 < theta < 40 degC, 0 < S < 40 psu,
    # 0 < p < 1e8 Pa. This is MOM6's EOS_DEFAULT (MOM_EOS.F90: EOS_DEFAULT =
    # EOS_WRIGHT_FULL_STRING).
    "wright_full": (
        7.133718e-4, 2.724670e-7, -1.646582e-7,
        5.613770e8, 3.600337e6, -3.727194e4, 1.660557e2, 6.844158e5, -8.389457e3,
        1.609893e5, 8.427815e2, -6.931554, 3.869318e-2, -1.664201e2, -2.765195,
    ),
    # Fit to UNESCO over the restricted range: -2 < theta < 30 degC,
    # 28 < S < 38 psu, 0 < p < 5e7 Pa.
    "wright_red": (
        7.057924e-4, 3.480336e-7, -1.112733e-7,
        5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3,
        1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464,
    ),
}

#: Validity ranges as stated in the MOM6 source, as (theta_min, theta_max,
#: salt_min, salt_max, pressure_min, pressure_max). Used only for warnings.
_WRIGHT_VALID_RANGE = {
    "wright_full": (-2.0, 40.0, 0.0, 40.0, 0.0, 1.0e8),
    "wright_red": (-2.0, 30.0, 28.0, 38.0, 0.0, 5.0e7),
}

#: MOM6 EQN_OF_STATE default, MOM_EOS.F90 ``EOS_DEFAULT``.
DEFAULT_EOS = "wright_full"

#: MOM6 ``RHO_0`` default, MOM_verticalGrid.F90 [kg m-3].
RHO_0 = 1035.0

#: MOM6 ``C_P`` default, MOM.F90 [J kg-1 K-1]. This is the TEOS-10 conservative
#: temperature value, which is what MOM6 uses regardless of whether its
#: temperature variable is potential or conservative temperature.
HEAT_CAPACITY = 3991.86795711963

#: Gravitational acceleration used for the Boussinesq pressure estimate [m s-2].
GRAVITY = 9.8


def available_eos() -> tuple:
    """Names of the implemented equations of state."""
    return tuple(_WRIGHT_COEFFICIENTS.keys())


def wright_density(theta, salt, pressure, eos=DEFAULT_EOS, check_range=False):
    """In situ density of seawater following Wright (1997).

    Ported from MOM6 ``src/equation_of_state/MOM_EOS_Wright_full.F90``, function
    ``density_elem_Wright_full``. The ``wright_red`` variant comes from
    ``MOM_EOS_Wright_red.F90``; the two share this expression and differ only in
    their coefficients::

        al0    = a0 + (a1*T + a2*S)
        p0     = b0 + ( b4*S + T * (b1 + (T*(b2 + b3*T) + b5*S)) )
        lambda = c0 + ( c4*S + T * (c1 + (T*(c2 + c3*T) + c5*S)) )
        rho    = (p + p0) / (lambda + al0*(p + p0))

    Passing ``pressure=0`` gives potential density referenced to the surface,
    which is the quantity MOM6 uses to diagnose mixed layer depth.

    This is an elementwise operation, so it broadcasts over any shapes and stays
    lazy if handed dask arrays.

    Args:
        theta (array_like): potential temperature relative to the surface [degC]
        salt (array_like): practical salinity [psu]
        pressure (array_like): pressure [Pa]
        eos (str, optional): one of :func:`available_eos`. Defaults to
            ``"wright_full"``, matching MOM6's ``EOS_DEFAULT``.
        check_range (bool, optional): if True, log a warning when inputs fall
            outside the range the fit was made over. Off by default because it
            forces evaluation of lazy arrays.

    Returns:
        array_like: in situ density [kg m-3]
    """
    if eos not in _WRIGHT_COEFFICIENTS:
        raise NotImplementedError(
            f"wright_density: unrecognized equation of state {eos!r}, "
            f"implemented options are {available_eos()}"
        )

    a0, a1, a2, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5 = _WRIGHT_COEFFICIENTS[eos]

    if check_range:
        _warn_outside_range(theta, salt, pressure, eos)

    al0 = a0 + (a1 * theta + a2 * salt)
    p0 = b0 + (b4 * salt + theta * (b1 + (theta * (b2 + b3 * theta) + b5 * salt)))
    lam = c0 + (c4 * salt + theta * (c1 + (theta * (c2 + c3 * theta) + c5 * salt)))
    return (pressure + p0) / (lam + al0 * (pressure + p0))


def _warn_outside_range(theta, salt, pressure, eos) -> None:
    """Log a warning if inputs stray outside the range the Wright fit covers."""

    tmin, tmax, smin, smax, pmin, pmax = _WRIGHT_VALID_RANGE[eos]
    for name, values, lo, hi in (
        ("potential temperature", theta, tmin, tmax),
        ("salinity", salt, smin, smax),
        ("pressure", pressure, pmin, pmax),
    ):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        low, high = np.nanmin(values), np.nanmax(values)
        if low < lo or high > hi:
            logger.warning(
                f"wright_density: {name} range [{low:.4g}, {high:.4g}] falls outside "
                f"the {eos} fit range [{lo:g}, {hi:g}]; densities there are extrapolated"
            )


def hydrostatic_pressure(depth, rho0=RHO_0, gravity=GRAVITY):
    """Boussinesq hydrostatic pressure at a given depth.

    ``p = rho0 * g * z``, the same approximation MOM6 uses to build the reference
    pressure passed to the equation of state in Boussinesq mode.

    Args:
        depth (array_like): depth below the surface, positive down [m]
        rho0 (float, optional): reference density [kg m-3]
        gravity (float, optional): gravitational acceleration [m s-2]

    Returns:
        array_like: pressure [Pa]
    """
    return rho0 * gravity * np.asarray(depth, dtype=float)


def interfaces_from_centers(depth):
    """Reconstruct layer interface depths from layer center depths.

    For a z-coordinate the centers are interface midpoints, so with the surface
    pinned at zero the interfaces follow from::

        z_i[0]   = 0
        z_i[k+1] = 2*z_l[k] - z_i[k]

    This lets the diagnostics work from the ``level`` coordinate alone, rather
    than depending on a hard coded table of interface depths, so it applies to
    any z-coordinate ocean source rather than just replay.

    Args:
        depth (array_like): layer center depths, positive down, increasing [m]

    Returns:
        numpy.ndarray: ``len(depth) + 1`` interface depths starting at 0 [m]

    Raises:
        ValueError: if the reconstruction is not strictly increasing, which means
            the input is not a z-coordinate whose centers bisect its interfaces.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 1:
        raise ValueError(
            f"interfaces_from_centers: expected a 1D array of layer centers, got shape {depth.shape}"
        )
    if depth.size == 0:
        raise ValueError("interfaces_from_centers: got an empty array of layer centers")
    if not np.all(np.diff(depth) > 0):
        raise ValueError(
            "interfaces_from_centers: layer centers must be strictly increasing with depth"
        )

    interfaces = np.empty(depth.size + 1, dtype=float)
    interfaces[0] = 0.0
    for k in range(depth.size):
        interfaces[k + 1] = 2.0 * depth[k] - interfaces[k]

    if not np.all(np.diff(interfaces) > 0):
        bad = int(np.argmin(np.diff(interfaces)))
        raise ValueError(
            "interfaces_from_centers: reconstructed interfaces are not strictly increasing "
            f"(first failure between interface {bad} at {interfaces[bad]:.4g} m and "
            f"{bad + 1} at {interfaces[bad + 1]:.4g} m). The 'level' coordinate does not look "
            "like a z-coordinate whose centers bisect its interfaces; pass explicit interfaces "
            "instead."
        )
    return interfaces


def _leading_valid_mask(*arrays):
    """Mask of levels forming the unbroken run of valid data from the surface.

    MOM6 columns are filled below the bathymetry, so the first invalid level in a
    column marks the ocean bottom. Anything below it is excluded even if isolated
    valid values appear deeper, which matches how
    :func:`ufs2arco.transforms.vertical_regrid.fv_vertical_regrid_ocean` already
    infers the bottom from NaNs in the temperature field.

    Args:
        *arrays (numpy.ndarray): arrays shaped ``(..., nz)``

    Returns:
        numpy.ndarray: boolean array shaped ``(..., nz)``
    """
    finite = np.ones(np.broadcast_shapes(*(a.shape for a in arrays)), dtype=bool)
    for array in arrays:
        finite &= np.isfinite(array)
    return np.cumprod(finite, axis=-1).astype(bool)


def mixed_layer_depth_by_density_difference(
    theta,
    salt,
    depth,
    density_diff=0.03,
    ref_pressure=0.0,
    eos=DEFAULT_EOS,
    pathological_to_nan=True,
):
    """Mixed layer depth from a potential density threshold.

    Ported from MOM6 ``src/diagnostics/MOM_diagnose_MLD.F90``, subroutine
    ``diagnoseMLDbyDensityDifference``, for the ``pRef_MLD == 0`` branch. That is
    the branch MOM6 takes by default, since ``HREF_FOR_MLD`` defaults to 0
    (``MOM_diabatic_driver.F90``).

    MOM6 scans down the column tracking the depth of each layer center, and takes
    the mixed layer depth to be where the potential density first exceeds the
    near-surface value by ``density_diff``, interpolated linearly between the two
    bracketing layer centers::

        rhoSurf = rho(T[0], S[0], ref_pressure)
        for k in 1..nz-1:
            drho_k = rho(T[k], S[k], ref_pressure) - rhoSurf
            if not found and (drho_k - drho_km1) > 0
                        and drho_km1 < density_diff <= drho_k:
                a   = (density_diff - drho_km1) / (drho_k - drho_km1)
                mld = a*depth[k] + (1 - a)*depth[k-1]
        if not found and drho_last < density_diff:
            mld = depth of deepest layer center      # mixing goes to the bottom

    The ``(drho_k - drho_km1) > 0`` test is what keeps a density inversion from
    triggering a spurious crossing, and the bottom fallback is conditional, so a
    column can legitimately leave MOM6's loop with no mixed layer depth set.

    Because the replay ocean lives on fixed z-levels, MOM6's running layer-center
    depth is exactly the ``level`` coordinate, so no layer thickness is needed.

    Two deliberate deviations from MOM6:

    1. **Land and bathymetry.** MOM6 masks land separately, so a fill-valued
       column there would fall through to its ``MLD = 0`` initialization. Here a
       column with no valid surface level returns NaN, and the scan stops at the
       first invalid level so the bottom fallback uses the deepest *valid* layer
       center.
    2. **Pathological zeros.** Columns that exit MOM6's loop with nothing set keep
       its ``0.0``, which would be a silent outlier in a training dataset. With
       ``pathological_to_nan=True`` (the default) they become NaN instead; set it
       False to reproduce MOM6 exactly. Either way the count is logged.

    Args:
        theta (array_like): potential temperature shaped ``(..., nz)`` [degC]
        salt (array_like): practical salinity shaped ``(..., nz)`` [psu]
        depth (array_like): layer center depths shaped ``(nz,)``, positive down [m]
        density_diff (float, optional): density threshold [kg m-3]. MOM6 uses
            0.03 for ``MLD_003`` and 0.125 for ``MLD_0125``.
        ref_pressure (float, optional): reference pressure for potential density
            [Pa]. Only 0.0 is supported, matching MOM6's default.
        eos (str, optional): equation of state, see :func:`available_eos`
        pathological_to_nan (bool, optional): see above

    Returns:
        numpy.ndarray: mixed layer depth shaped ``(...)`` [m]

    Raises:
        NotImplementedError: if ``ref_pressure`` is nonzero. MOM6 supports a
            nonzero ``HREF_FOR_MLD`` by interpolating the reference density to
            that depth, which is not implemented here.
    """
    if ref_pressure != 0.0:
        raise NotImplementedError(
            "mixed_layer_depth_by_density_difference: only ref_pressure=0 is implemented. "
            "MOM6 supports a nonzero HREF_FOR_MLD by interpolating the reference density to "
            f"that depth, which is not ported here; got ref_pressure={ref_pressure}"
        )

    theta = np.asarray(theta, dtype=float)
    salt = np.asarray(salt, dtype=float)
    depth = np.asarray(depth, dtype=float)

    if depth.ndim != 1:
        raise ValueError(
            f"mixed_layer_depth_by_density_difference: depth must be 1D, got shape {depth.shape}"
        )
    nz = depth.size
    for name, array in (("theta", theta), ("salt", salt)):
        if array.shape[-1] != nz:
            raise ValueError(
                f"mixed_layer_depth_by_density_difference: {name} has {array.shape[-1]} levels "
                f"on its last axis but depth has {nz}"
            )
    if nz < 2:
        raise ValueError(
            "mixed_layer_depth_by_density_difference: need at least 2 vertical levels, "
            f"got {nz}"
        )

    valid = _leading_valid_mask(theta, salt)
    rho = wright_density(theta, salt, ref_pressure, eos=eos)

    # rhoSurf is the density of the topmost layer, per the pRef_MLD == 0 branch
    delta_rho = rho - rho[..., :1]
    # Below the bottom the difference is meaningless. A large negative sentinel
    # can never satisfy the crossing test, which stops the scan at the
    # bathymetry. A finite value is used rather than -inf so that differencing
    # across the boundary stays well defined.
    below_bottom = -1.0e30
    delta_rho = np.where(valid, delta_rho, below_bottom)

    ddrho = np.diff(delta_rho, axis=-1)
    crossing = (
        (ddrho > 0.0)
        & (delta_rho[..., :-1] < density_diff)
        & (delta_rho[..., 1:] >= density_diff)
    )

    found = crossing.any(axis=-1)
    # argmax gives the first True; where nothing is True it gives 0, which the
    # `found` mask discards below.
    k = np.argmax(crossing, axis=-1)

    drho_km1 = np.take_along_axis(delta_rho[..., :-1], k[..., None], axis=-1)[..., 0]
    drho_k = np.take_along_axis(delta_rho[..., 1:], k[..., None], axis=-1)[..., 0]
    # Strictly positive wherever `found`, by the ddrho > 0 term in `crossing`.
    denominator = np.where(found, drho_k - drho_km1, 1.0)
    a_fac = (density_diff - drho_km1) / denominator
    interpolated = a_fac * depth[k + 1] + (1.0 - a_fac) * depth[k]

    # Bottom fallback: MOM6 sets the mixed layer depth to the deepest layer
    # center reached, but only when the deepest density difference is still below
    # the threshold.
    n_valid = valid.sum(axis=-1)
    is_ocean = n_valid > 0
    deepest_index = np.clip(n_valid - 1, 0, nz - 1)
    deepest_delta_rho = np.take_along_axis(delta_rho, deepest_index[..., None], axis=-1)[..., 0]
    reaches_bottom = (~found) & is_ocean & (deepest_delta_rho < density_diff)

    mld = np.full(found.shape, 0.0, dtype=float)
    mld = np.where(found, interpolated, mld)
    mld = np.where(reaches_bottom, depth[deepest_index], mld)

    pathological = (~found) & (~reaches_bottom) & is_ocean
    n_pathological = int(pathological.sum())
    if n_pathological > 0:
        disposition = "set to NaN" if pathological_to_nan else "left at 0.0, as MOM6 does"
        logger.warning(
            f"mixed_layer_depth_by_density_difference: {n_pathological} ocean column(s) "
            f"resolved to no mixed layer depth and no bottom fallback; {disposition}. "
            "This happens where the density profile is non-monotonic right at the threshold."
        )
        if pathological_to_nan:
            mld = np.where(pathological, np.nan, mld)

    # Land, and anything with no valid surface level
    mld = np.where(is_ocean, mld, np.nan)
    return mld


def ocean_heat_content(
    theta,
    interfaces,
    max_depth=None,
    rho0=RHO_0,
    heat_capacity=HEAT_CAPACITY,
    require_full_depth=True,
):
    """Depth-integrated ocean heat content.

    The Boussinesq form MOM6 uses, ``OHC = rho0 * Cp * integral(theta dz)``,
    evaluated as a finite volume sum with partial-cell weighting on the layer
    that straddles ``max_depth``::

        w_k = clip((max_depth - z_i[k]) / dz_k, 0, 1)
        OHC = rho0 * Cp * sum_k w_k * dz_k * theta_k

    Levels below the bathymetry contribute nothing. Note that ``heat_capacity``
    defaults to MOM6's ``C_P``, which upstream is documented as the TEOS-10
    conservative temperature value; MOM6 applies it whether its temperature
    variable is potential or conservative temperature, and so does this. Set it
    explicitly to use a potential temperature convention instead.

    Args:
        theta (array_like): potential temperature shaped ``(..., nz)`` [degC]
        interfaces (array_like): layer interface depths shaped ``(nz + 1,)``,
            positive down and starting at the surface [m]. See
            :func:`interfaces_from_centers`.
        max_depth (float, optional): integrate from the surface to this depth
            [m]. None integrates the whole valid column.
        rho0 (float, optional): Boussinesq reference density [kg m-3]
        heat_capacity (float, optional): heat capacity of seawater [J kg-1 K-1]
        require_full_depth (bool, optional): if True, return NaN
            where the water column does not reach the ``max_depth``. Set
            False to accept partial columns (the default).

    Returns:
        numpy.ndarray: heat content shaped ``(...)`` [J m-2]
    """
    theta = np.asarray(theta, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)

    if interfaces.ndim != 1:
        raise ValueError(
            f"ocean_heat_content: interfaces must be 1D, got shape {interfaces.shape}"
        )
    nz = theta.shape[-1]
    if interfaces.size != nz + 1:
        raise ValueError(
            f"ocean_heat_content: got {interfaces.size} interfaces for {nz} levels, "
            f"expected {nz + 1}"
        )
    if not np.all(np.diff(interfaces) > 0):
        raise ValueError("ocean_heat_content: interfaces must be strictly increasing with depth")

    valid = _leading_valid_mask(theta)
    thickness = np.diff(interfaces)

    if max_depth is None:
        weights = np.ones(nz, dtype=float)
    else:
        if max_depth <= 0:
            raise ValueError(f"ocean_heat_content: max_depth must be positive, got {max_depth}")
        weights = np.clip((max_depth - interfaces[:-1]) / thickness, 0.0, 1.0)

    contribution = np.where(valid, np.nan_to_num(theta, nan=0.0) * (weights * thickness), 0.0)
    result = rho0 * heat_capacity * contribution.sum(axis=-1)

    # Depth of the deepest valid interface in each column
    n_valid = valid.sum(axis=-1)
    is_ocean = n_valid > 0
    column_bottom = interfaces[n_valid]

    if max_depth is not None and require_full_depth:
        is_ocean = is_ocean & (column_bottom >= max_depth)

    return np.where(is_ocean, result, np.nan)