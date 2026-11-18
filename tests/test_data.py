"""Time-zone handling in the data pipeline, on tiny hand-made source files with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rl import data as D


def _ercot_rows(rows):
    return pd.DataFrame(rows, columns=["Delivery Date", "Delivery Hour", "Delivery Interval", "Repeated Hour Flag",
                                       "Settlement Point Name", "Settlement Point Type", "Settlement Point Price"])


def test_ercot_rtm_to_utc(tmp_path, monkeypatch):
    rows = [
        ("01/01/2018", 1, 1, "N", "HB_HUBAVG", "SH", 10.0),   # 00:00 CST            -> 06:00 UTC
        ("01/01/2018", 24, 4, "N", "HB_HUBAVG", "SH", 11.0),  # 23:45 CST            -> 05:45 UTC next day
        ("03/11/2018", 4, 1, "N", "HB_HUBAVG", "SH", 12.0),   # spring forward: ERCOT skips HE 3; HE 4 = 03:00 CDT -> 08:00 UTC
        ("07/01/2018", 1, 1, "N", "HB_HUBAVG", "SH", 13.0),   # 00:00 CDT            -> 05:00 UTC
        ("11/04/2018", 2, 1, "N", "HB_HUBAVG", "SH", 14.0),   # first 01:00, CDT     -> 06:00 UTC
        ("11/04/2018", 2, 1, "Y", "HB_HUBAVG", "SH", 15.0),   # repeated 01:00, CST  -> 07:00 UTC
        ("11/04/2018", 2, 1, "N", "HB_HOUSTON", "LZ", 99.0),  # other settlement point, dropped
    ]
    path = tmp_path / "rtm.xlsx"
    with pd.ExcelWriter(path) as w:  # the real file has one sheet per month
        _ercot_rows(rows[:2]).to_excel(w, sheet_name="Jan", index=False)
        _ercot_rows(rows[2:]).to_excel(w, sheet_name="Mar-Nov", index=False)
    monkeypatch.setattr(D, "ERCOT_XLSX", path)
    p = D.load_prices()
    assert p.index.is_monotonic_increasing
    expected = {"2018-01-01 06:00": 0.010, "2018-01-02 05:45": 0.011, "2018-03-11 08:00": 0.012,
                "2018-07-01 05:00": 0.013, "2018-11-04 06:00": 0.014, "2018-11-04 07:00": 0.015}
    assert len(p) == len(expected)
    for ts, price in expected.items():
        assert p[pd.Timestamp(ts, tz="UTC")] == pytest.approx(price)  # $/MWh -> $/kWh


def test_resstock_end_of_interval_lst_to_utc(tmp_path, monkeypatch):
    bldg = 1
    (tmp_path / "resstock").mkdir()
    # ResStock: local standard time all year, timestamp at the END of the interval, kWh per 15 minutes
    pd.DataFrame({"timestamp": pd.to_datetime(["2018-01-01 00:15", "2018-07-01 00:15"]),
                  "out.electricity.total.energy_consumption": [0.5, 1.0]}).to_parquet(
        tmp_path / "resstock" / f"{bldg}-0.parquet")
    monkeypatch.setattr(D, "RAW", tmp_path)
    load = D.load_household(bldg)
    # start 00:00 UTC-6 = 06:00 UTC in both January and July (no DST shift), and kWh/15 min -> kW
    assert load[pd.Timestamp("2018-01-01 06:00", tz="UTC")] == pytest.approx(2.0)
    assert load[pd.Timestamp("2018-07-01 06:00", tz="UTC")] == pytest.approx(4.0)


def test_split_is_disjoint_and_covers_the_year():
    days = np.arange(365)
    s = D.split_days(days)
    allocated = np.concatenate(list(s.values()))
    assert len(allocated) == len(np.unique(allocated)) == 365
    assert set(s["test"]) == {d for d in days if (d // 7) % 4 == 3}


@pytest.mark.skipif(not (D.PROCESSED / "austin_2018.parquet").exists(), reason="dataset not built")
def test_built_dataset_has_96_steps_every_day():
    df = D.load_dataset()
    assert len(df) == 365 * 96 and not df[["price", "load", "solar"]].isna().any().any()
    assert (df.groupby("day").size() == 96).all()
    assert (df.groupby("day")["step"].apply(lambda s: list(s) == list(range(96)))).all()
    # solar is zero at local midnight and peaks around solar noon (about 12:30 local standard time in Austin)
    by_step = df.groupby("step")["solar"].mean()
    assert by_step.iloc[0] == 0 and 46 <= by_step.idxmax() <= 54


def test_ercot_rejects_the_skipped_spring_forward_hour(tmp_path, monkeypatch):
    path = tmp_path / "rtm.xlsx"
    _ercot_rows([("03/11/2018", 3, 1, "N", "HB_HUBAVG", "SH", 1.0)]).to_excel(path, index=False)
    monkeypatch.setattr(D, "ERCOT_XLSX", path)
    with pytest.raises(Exception, match="nonexistent"):
        D.load_prices()
