"""Download the raw 2018 inputs into data/raw/, then build the dataset.

    .venv/bin/python scripts/download_data.py            # download (skips files already present) and build
    .venv/bin/python scripts/download_data.py --no-build # download only

Sources (all public, no login):
- ERCOT "Historical RTM Load Zone and Hub Prices" (report type 13061, NP6-785-ER), 2018 file
- ERCOT "Historical DAM Load Zone and Hub Prices" (report type 13060, NP4-180-ER), 2018 file (only needed for the
  day-ahead forecast features)
- CityLearn's Austin AMY2018 weather file (tx_travis_county_neighborhood/weather.epw)
- The ResStock household load is fetched by battery_rl.data itself when the dataset is built.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile

from battery_rl.data import EPW, ERCOT_XLSX, RAW

ERCOT_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rt}"
ERCOT_GET = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc}"
EPW_URL = ("https://raw.githubusercontent.com/intelligent-environments-lab/CityLearn/master/"
           "data/datasets/tx_travis_county_neighborhood/weather.epw")
ERCOT_REPORTS = {13061: "RTMLZHBSPP_2018", 13060: "DAMLZHBSPP_2018"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "battery-rl-course-project"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def ercot_year_file(report_type: int, friendly_name: str) -> None:
    out_dir = RAW / "ercot"
    if any(out_dir.glob(f"*{friendly_name}.xlsx")):
        print(f"have {friendly_name}")
        return
    docs = json.loads(fetch(ERCOT_LIST.format(rt=report_type)))["ListDocsByRptTypeRes"]["DocumentList"]
    match = [d["Document"] for d in docs if d["Document"]["FriendlyName"] == friendly_name]
    if not match:
        raise RuntimeError(f"ERCOT report {report_type}: no document named {friendly_name}")
    print(f"downloading {match[0]['ConstructedName']} ({int(match[0]['ContentSize']) / 1e6:.0f} MB)")
    with zipfile.ZipFile(io.BytesIO(fetch(ERCOT_GET.format(doc=match[0]["DocID"])))) as z:
        out_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true", help="only download, do not build the processed dataset")
    args = ap.parse_args()

    for rt, name in ERCOT_REPORTS.items():
        ercot_year_file(rt, name)
    if not ERCOT_XLSX.exists():
        raise RuntimeError(f"expected {ERCOT_XLSX} after download")
    if EPW.exists():
        print("have weather.epw")
    else:
        EPW.parent.mkdir(parents=True, exist_ok=True)
        EPW.write_bytes(fetch(EPW_URL))
        print(f"downloaded {EPW}")

    if not args.no_build:
        from battery_rl.data import main as build

        build()


if __name__ == "__main__":
    main()
