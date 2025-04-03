# Integration tests

## `test_datasets.py`

Purpose is to test different combos of sources and targets

Test everything with
```
pytest test_datasets.py
```

Test a specific combo, e.g. `source="gefs"` and `target="anemoi"` with
```
pytest -k "test_this_combo[gefs-anemoi]"
```
