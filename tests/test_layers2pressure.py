import numpy as np
import xarray as xr

from ufs2arco.layers2pressure import Layers2Pressure


def _l2p():
    ak = np.array([0.0, 0.0, 0.0])
    bk = np.array([0.1, 0.5, 1.0])
    return Layers2Pressure(ak=ak, bk=bk, level_name="level", interface_name="interface")


def test_init_requires_both_ak_and_bk():
    try:
        Layers2Pressure(ak=np.array([0.0, 1.0]))
        raise AssertionError("Expected TypeError when only ak is provided")
    except TypeError:
        pass


def test_calc_pressure_interfaces_and_thickness_shapes():
    l2p = _l2p()
    pressfc = xr.DataArray([100000.0], dims=("point",))
    prsi = l2p.calc_pressure_interfaces(pressfc)
    dpres = l2p.calc_pressure_thickness(prsi)
    assert "interface" in prsi.dims
    assert "level" in dpres.dims
    assert len(dpres["level"]) == len(l2p.pfull)


def test_calc_delz_uses_qmin_floor():
    l2p = _l2p()
    pressfc = xr.DataArray([100000.0], dims=("point",))
    temp = xr.DataArray([[280.0, 270.0]], dims=("point", "level"), coords={"level": l2p.pfull.values})
    spfh = xr.DataArray([[-1e-6, 1e-12]], dims=("point", "level"), coords={"level": l2p.pfull.values})
    delz = l2p.calc_delz(pressfc=pressfc, temp=temp, spfh=spfh)
    assert np.isfinite(delz.values).all()


def test_get_interp_coefficients_bounds_factor_and_denominator():
    l2p = _l2p()
    prsl = xr.DataArray([[300.0, 800.0]], dims=("point", "level"), coords={"level": l2p.pfull.values})
    cds = l2p.get_interp_coefficients(pstar=500.0, prsl=prsl)
    assert float(cds["denominator"].min()) >= 1e-6
    assert float(cds["factor"].max()) <= 10.0
    assert float(cds["factor"].min()) >= -10.0


def test_interp2pressure_returns_nan_when_no_valid_bracketing_levels():
    l2p = _l2p()
    xda = xr.DataArray([[1.0, 2.0]], dims=("point", "level"), coords={"level": l2p.pfull.values})
    prsl = xr.DataArray([[300.0, 800.0]], dims=("point", "level"), coords={"level": l2p.pfull.values})
    out = l2p.interp2pressure(xda=xda, pstar=50.0, prsl=prsl)
    assert np.isnan(out.values).all()
