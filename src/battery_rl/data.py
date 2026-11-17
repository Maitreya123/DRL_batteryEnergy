"""Build the 2018 Austin dataset: ERCOT price, household load and rooftop solar on one 15-minute UTC grid.

Time conventions of the raw sources, all converted to UTC before merging:
- ERCOT RTM (NP6-785-ER): Central *prevailing* time (with DST). "Delivery Hour" is hour-ending (1..24),
  "Delivery Interval" is 1..4 within the hour, and "Repeated Hour Flag" = "Y" marks the second copy of
  the repeated hour when DST ends in November.
- ResStock AMY2018: local *standard* time (UTC-6, no DST), timestamp marks the END of each 15-minute interval.
- EPW weather (Camp Mabry, AMY2018): local standard time, hourly, hour-ending values.

Each row of the output is labelled by the START of its 15-minute interval, in UTC.
Days are defined in local standard time (UTC-6), so every day has exactly 96 steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

LST = "Etc/GMT+6"  # local standard time for Texas (UTC-6, never shifts)
CENTRAL = "America/Chicago"  # prevailing Central time (with DST), used by ERCOT
STEP = pd.Timedelta(minutes=15)

ERCOT_XLSX = RAW / "ercot" / "rpt.00013061.0000000000000000.RTMLZHBSPP_2018.xlsx"
EPW = RAW / "citylearn" / "weather.epw"
RESSTOCK_URL = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
    "2021/resstock_amy2018_release_1/timeseries_individual_buildings/by_county/upgrade=0/county=G4804530/{bldg}-0.parquet"
)
DEFAULT_BUILDING = 102188  # Travis County single-family home; also in CityLearn's tx_travis_county_neighborhood


def load_prices(settlement_point: str = "HB_HUBAVG") -> pd.Series:
    """ERCOT real-time price in $/kWh, indexed by interval start in UTC."""
    sheets = pd.read_excel(ERCOT_XLSX, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df[df["Settlement Point Name"] == settlement_point].copy()

    date = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")
    start_local = (
        date
        + pd.to_timedelta(df["Delivery Hour"].astype(int) - 1, unit="h")
        + pd.to_timedelta((df["Delivery Interval"].astype(int) - 1) * 15, unit="min")
    )
    # In the repeated (fall-back) hour, the first copy is still daylight time (flag N), the second is standard (Y).
    is_dst_for_ambiguous = (df["Repeated Hour Flag"].astype(str).str.upper() != "Y").to_numpy()
    start_utc = (
        pd.DatetimeIndex(start_local)
        .tz_localize(CENTRAL, ambiguous=is_dst_for_ambiguous, nonexistent="raise")
        .tz_convert("UTC")
    )
    price = pd.Series(df["Settlement Point Price"].to_numpy(float) / 1000.0, index=start_utc, name="price")
    price = price.sort_index()
    if price.index.has_duplicates:
        raise ValueError("duplicate ERCOT timestamps after UTC conversion")
    return price


def load_household(bldg: int = DEFAULT_BUILDING) -> pd.Series:
    """Total household electricity demand as average kW over each 15-minute interval, UTC interval start."""
    path = RAW / "resstock" / f"{bldg}-0.parquet"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RESSTOCK_URL.format(bldg=bldg), path)
    df = pd.read_parquet(path, columns=["timestamp", "out.electricity.total.energy_consumption"])
    end_local = pd.DatetimeIndex(df["timestamp"]).tz_localize(LST)
    start_utc = (end_local - STEP).tz_convert("UTC")
    kwh = df["out.electricity.total.energy_consumption"].to_numpy(float)
    return pd.Series(kwh / STEP.total_seconds() * 3600.0, index=start_utc, name="load")  # kWh per 15 min -> kW


def load_solar(kw_dc: float = 5.0, tilt: float = 25.0, azimuth: float = 180.0) -> pd.Series:
    """AC output (kW) of a rooftop PV system from 2018 Camp Mabry weather, via pvlib's PVWatts model.

    Each hourly EPW row is evaluated at a single representative time, then the hourly power is linearly
    interpolated to the midpoints of the 15-minute intervals.

    pvlib labels the first EPW row 00:00 (Hour 1 minus one hour). The representative time within the hour
    was fitted, not assumed: correlating the year's measured GHI with pvlib's clear-sky GHI peaks at a
    shift of +10 to +15 minutes past pvlib's label (r = 0.8937 at +10, 0.8925 at 0, 0.8881 at +30).
    """
    import pvlib

    weather, meta = pvlib.iotools.read_epw(EPW, coerce_year=2018)
    weather.index = weather.index + pd.Timedelta(minutes=15)
    loc = pvlib.location.Location(meta["latitude"], meta["longitude"], tz=LST, altitude=meta["altitude"])
    solpos = loc.get_solarposition(weather.index, temperature=weather["temp_air"])
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, azimuth, solpos["apparent_zenith"], solpos["azimuth"],
        weather["dni"], weather["ghi"], weather["dhi"], dni_extra=dni_extra, model="haydavies",
    )
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    cell_t = pvlib.temperature.sapm_cell(poa["poa_global"], weather["temp_air"], weather["wind_speed"], **params)
    dc = pvlib.pvsystem.pvwatts_dc(poa["poa_global"], cell_t, pdc0=kw_dc * 1000.0, gamma_pdc=-0.0037)
    dc = dc * (1 - 0.14)  # PVWatts default system losses (soiling, wiring, mismatch, ...)
    ac = pvlib.inverter.pvwatts(dc, pdc0=kw_dc * 1000.0 / 0.96, eta_inv_nom=0.96).clip(lower=0.0) / 1000.0

    hourly = pd.Series(ac.to_numpy(float), index=weather.index.tz_convert("UTC"))
    grid = pd.date_range("2018-01-01 00:00", "2018-12-31 23:45", freq=STEP, tz=LST).tz_convert("UTC")
    mid = grid + STEP / 2
    x = hourly.index.asi8.astype(float)
    vals = np.interp(mid.asi8.astype(float), x, hourly.to_numpy(), left=0.0, right=0.0)
    return pd.Series(np.clip(vals, 0.0, None), index=grid, name="solar")


def build(bldg: int = DEFAULT_BUILDING, settlement_point: str = "HB_HUBAVG") -> pd.DataFrame:
    """Merge all inputs onto the 2018 local-standard-time 15-minute grid (35,040 rows, 365 days x 96)."""
    grid = pd.date_range("2018-01-01 00:00", "2018-12-31 23:45", freq=STEP, tz=LST).tz_convert("UTC")
    price = load_prices(settlement_point).reindex(grid)
    load = load_household(bldg).reindex(grid)
    solar = load_solar().reindex(grid)
    df = pd.DataFrame({"price": price, "load": load, "solar": solar}, index=grid)
    df.index.name = "start_utc"

    # ERCOT's 2018 file starts at 00:00 Central prevailing time on Jan 1, which is 00:00 LST, and ends at
    # 23:45 Dec 31, so the grid is fully covered; any gap would be a conversion bug, not missing data.
    missing = df.isna().sum()
    if missing.any():
        raise ValueError(f"missing values after merge:\n{missing[missing > 0]}")

    local = df.index.tz_convert(LST)
    df["day"] = (local.normalize() - pd.Timestamp("2018-01-01", tz=LST)).days
    df["step"] = (local.hour * 4 + local.minute // 15).astype(int)
    df["dow"] = local.day_of_week.astype(int)
    return df


def split_days(days: np.ndarray) -> dict[str, np.ndarray]:
    """Week-based split so every season appears in each set.

    Weeks are 7-day blocks from Jan 1. Test: one week in every four (week % 4 == 3), as in the proposal.
    Validation (for hyperparameter sweeps): week % 8 == 1. Train: everything else (~62% of days).
    """
    week = days // 7
    test = week % 4 == 3
    val = (week % 8 == 1) & ~test
    train = ~test & ~val
    return {"train": days[train], "val": days[val], "test": days[test]}


def load_dataset() -> pd.DataFrame:
    path = PROCESSED / "austin_2018.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m battery_rl.data` first")
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", type=int, default=DEFAULT_BUILDING)
    ap.add_argument("--settlement-point", default="HB_HUBAVG")
    args = ap.parse_args()
    df = build(args.building, args.settlement_point)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "austin_2018.parquet"
    df.to_parquet(out)
    print(f"wrote {out}: {len(df)} rows, {df['day'].nunique()} days")
    print(df[["price", "load", "solar"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
