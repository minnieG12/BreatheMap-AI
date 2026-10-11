"""Repair station identifiers in the already-processed Delhi AQI dataset.

The source export repeats a generic source ID across stations. This script
creates stable station IDs from station_name and refreshes the station summary
and quality report without re-downloading/reprocessing the raw dataset.
"""
from pathlib import Path
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LONG_FILE = PROCESSED / "delhi_aqi_long.csv"
SUMMARY_FILE = PROCESSED / "delhi_station_summary.csv"
REPORT_FILE = PROCESSED / "data_report.txt"


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unknown_station"


def main() -> int:
    if not LONG_FILE.exists():
        print(f"Missing {LONG_FILE}. Run scripts/prepare_delhi_data.py first.")
        return 1

    df = pd.read_csv(LONG_FILE, parse_dates=["timestamp"])
    required = {"timestamp", "station_id", "state", "city", "station_name", "aqi"}
    missing = required.difference(df.columns)
    if missing:
        print(f"Unexpected columns; missing: {sorted(missing)}")
        return 2

    df["station_name"] = df["station_name"].astype("string").str.strip()
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df = df.dropna(subset=["timestamp", "station_name", "aqi"])
    df = df[(df["aqi"] >= 0) & (df["aqi"] <= 500)].copy()

    # Preserve the source-provided value for traceability, but don't use the
    # generic, repeated field as the unique station key.
    if "source_station_id" not in df.columns:
        df["source_station_id"] = df["station_id"].astype("string")
    df["station_id"] = df["station_name"].map(lambda x: "delhi_" + slugify(x))

    # Deduplicate exact station-time records, keeping the last row.
    df = df.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"], keep="last"
    )
    df = df[["timestamp", "station_id", "state", "city", "station_name", "aqi", "source_station_id"]]
    df.to_csv(LONG_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")

    stations = (
        df[["station_id", "state", "city", "station_name", "source_station_id"]]
        .drop_duplicates(subset=["station_id"])
        .sort_values("station_name")
    )
    stations.to_csv(SUMMARY_FILE, index=False)

    dates = df["timestamp"]
    report = (
        "BreatheMap AI — Phase 1 Data Report\n"
        "====================================\n"
        "Source: cpcb-aqi.csv.gz (already downloaded)\n"
        "Original source: https://github.com/Vonter/india-cpcb-aqi\n"
        "Original source license: ODbL-1.0 (see source repository).\n"
        "Start-date filter: data already filtered to 2023-01-01 onward\n"
        f"Valid station-hour observations exported: {len(df):,}\n"
        f"Unique station names: {df['station_name'].nunique():,}\n"
        f"Unique derived station IDs: {df['station_id'].nunique():,}\n"
        f"Duplicate station-timestamp pairs after cleanup: {df.duplicated(['station_id', 'timestamp']).sum():,}\n"
        f"AQI range: {df['aqi'].min():.0f} to {df['aqi'].max():.0f}\n"
        f"First timestamp: {dates.min()}\n"
        f"Last timestamp: {dates.max()}\n"
        "Station IDs are derived from station_name because the source ID was repeated across multiple station names.\n"
        "Note: this is historical AQI, not a live feed. Source time convention is retained without assigning a timezone.\n"
        "Do not describe unmeasured locations as observed; spatial estimates require coordinates and separate validation.\n"
    )
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("STATION ID FIX COMPLETE")
    print(report)
    print("Updated files:")
    print(f" - {LONG_FILE}")
    print(f" - {SUMMARY_FILE}")
    print(f" - {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
