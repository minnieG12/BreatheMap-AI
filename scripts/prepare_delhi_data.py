from pathlib import Path
import argparse
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "cpcb-aqi.csv.gz"
OUT_DIR = ROOT / "data" / "processed"
TIME_RE = re.compile(r"^\d{2}:00:00$")


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_col(columns, candidates):
    lookup = {normalized(c): c for c in columns}
    for candidate in candidates:
        if normalized(candidate) in lookup:
            return lookup[normalized(candidate)]
    return None


def main():
    parser = argparse.ArgumentParser(description="Filter and reshape Delhi hourly AQI data")
    parser.add_argument("--start-date", default="2023-01-01", help="Keep dates on/after this date, e.g. 2023-01-01")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"Dataset missing: {SOURCE}\nRun first: python scripts/download_dataset.py")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading headers...")
    headers = list(pd.read_csv(SOURCE, compression="gzip", nrows=0).columns)
    station_id_col = find_col(headers, ["Station ID", "station_id", "stationId"])
    state_col = find_col(headers, ["State", "state"])
    city_col = find_col(headers, ["City", "city"])
    name_col = find_col(headers, ["Station Name", "station_name", "station"])
    date_col = find_col(headers, ["Date", "date", "timestamp"])
    hour_cols = [c for c in headers if TIME_RE.match(str(c).strip())]

    required = {"Station ID": station_id_col, "State": state_col, "City": city_col, "Station Name": name_col, "Date": date_col}
    missing = [k for k, v in required.items() if v is None]
    if missing or len(hour_cols) < 12:
        print("Unexpected source schema.")
        print("Missing fields:", missing)
        print("Found columns:", headers[:40])
        print("Found hourly columns:", len(hour_cols))
        sys.exit(2)

    usecols = [station_id_col, state_col, city_col, name_col, date_col] + hour_cols
    out_long = OUT_DIR / "delhi_aqi_long.csv"
    out_summary = OUT_DIR / "delhi_station_summary.csv"
    out_report = OUT_DIR / "data_report.txt"
    if out_long.exists(): out_long.unlink()

    min_date = pd.Timestamp(args.start_date)
    total_rows = 0
    total_long = 0
    written = False
    all_stations = []
    min_seen = None
    max_seen = None
    print(f"Filtering Delhi station rows from {args.start_date} onward...")

    for chunk_no, chunk in enumerate(pd.read_csv(SOURCE, compression="gzip", usecols=usecols, chunksize=args.chunksize, low_memory=False), start=1):
        state = chunk[state_col].astype("string")
        city = chunk[city_col].astype("string")
        station_name = chunk[name_col].astype("string")
        mask = (
            state.str.contains("Delhi", case=False, na=False)
            | city.str.contains("Delhi", case=False, na=False)
            | station_name.str.contains("Delhi", case=False, na=False)
        )
        d = chunk.loc[mask].copy()
        if d.empty:
            if chunk_no % 5 == 0: print(f"  scanned {chunk_no * args.chunksize:,} source rows...")
            continue
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d[d[date_col].notna() & (d[date_col] >= min_date)]
        if d.empty:
            continue

        base_cols = [station_id_col, state_col, city_col, name_col, date_col]
        long = d.melt(id_vars=base_cols, value_vars=hour_cols, var_name="hour", value_name="aqi")
        long["aqi"] = pd.to_numeric(long["aqi"], errors="coerce")
        hour_num = long["hour"].str.slice(0, 2).astype(int)
        long["timestamp"] = long[date_col] + pd.to_timedelta(hour_num, unit="h")
        long = long.rename(columns={station_id_col: "source_station_id", state_col: "state", city_col: "city", name_col: "station_name"})
        # The source export may repeat a generic station ID across multiple
        # station names. Use a stable ID derived from station_name, and retain
        # the original source value separately for traceability.
        long["station_id"] = (
            "delhi_" + long["station_name"].astype("string").str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        )
        long = long[["timestamp", "station_id", "state", "city", "station_name", "aqi", "source_station_id"]]
        long = long.dropna(subset=["timestamp", "aqi"])
        long = long[(long["aqi"] >= 0) & (long["aqi"] <= 1000)]
        if long.empty:
            continue

        long.to_csv(out_long, mode="a", header=not written, index=False)
        written = True
        total_long += len(long)
        total_rows += len(d)
        all_stations.append(d[[station_id_col, state_col, city_col, name_col]].drop_duplicates())
        chunk_min, chunk_max = long["timestamp"].min(), long["timestamp"].max()
        min_seen = chunk_min if min_seen is None else min(min_seen, chunk_min)
        max_seen = chunk_max if max_seen is None else max(max_seen, chunk_max)
        print(f"  chunk {chunk_no}: retained {len(d):,} Delhi station-days; emitted {len(long):,} valid AQI hours")

    if not written:
        print("No Delhi data was found after the cutoff. Try --start-date 2020-01-01 and inspect source labels.")
        sys.exit(3)

    stations = pd.concat(all_stations, ignore_index=True).drop_duplicates()
    stations = stations.rename(columns={station_id_col: "source_station_id", state_col: "state", city_col: "city", name_col: "station_name"})
    stations["station_id"] = (
        "delhi_" + stations["station_name"].astype("string").str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    stations = stations[["station_id", "state", "city", "station_name", "source_station_id"]].sort_values("station_name")
    stations.to_csv(out_summary, index=False)
    report = (
        "BreatheMap AI — Phase 1 Data Report\n"
        "====================================\n"
        f"Source: {SOURCE.name}\n"
        "Original source: https://github.com/Vonter/india-cpcb-aqi\n"
        "Original source license: ODbL-1.0 (see source repository).\n"
        f"Start-date filter: {args.start_date}\n"
        f"Delhi station-day rows retained before melting: {total_rows:,}\n"
        f"Valid station-hour observations exported: {total_long:,}\n"
        f"Unique station names: {stations['station_name'].nunique()}\n"
        f"Unique derived station IDs: {stations['station_id'].nunique()}\n"
        "Station IDs are derived from station_name because the source ID may repeat across stations.\n"
        f"First timestamp: {min_seen}\n"
        f"Last timestamp: {max_seen}\n"
        "Note: this is historical AQI, not a live feed. Source time convention is retained without assigning a timezone.\n"
        "Do not describe unmeasured locations as observed; spatial estimates require coordinates and separate validation.\n"
    )
    out_report.write_text(report, encoding="utf-8")
    print("\nPHASE 1 SUCCESS")
    print(report)
    print(f"Outputs:\n - {out_long}\n - {out_summary}\n - {out_report}")


if __name__ == "__main__":
    main()
