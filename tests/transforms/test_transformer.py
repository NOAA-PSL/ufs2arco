import pytest
import xarray as xr

from ufs2arco.transforms.transformer import Transformer


def test_transformer_rejects_unknown_transform():
    with pytest.raises(
        NotImplementedError,
        match=r"transformations are not recognized or not implemented",
    ):
        Transformer(options={"not_real": {}})


def test_transformer_rejects_unknown_mapping():
    with pytest.raises(
        NotImplementedError,
        match=r"mappings are not recognized or not implemented",
    ):
        Transformer(options={"mappings": {"not_real": ["t2m"]}})


def test_transformer_applies_multiply_divide_and_mapping():
    xds = xr.Dataset({"t2m": ("x", [10.0, 12.0])}, coords={"x": [0, 1]})
    tr = Transformer(
        options={
            "multiply": {"t2m": 2},
            "divide": {"t2m": 4},
            "mappings": {"round": ["t2m"]},
        }
    )
    result = tr(xds)
    assert "t2m" not in result
    assert "round_t2m" in result
    assert result["round_t2m"].values.tolist() == [5.0, 6.0]
