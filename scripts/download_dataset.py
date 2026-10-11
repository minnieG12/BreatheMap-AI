from pathlib import Path
import requests

URL = "https://raw.githubusercontent.com/Vonter/india-cpcb-aqi/main/data/cpcb-aqi.csv.gz"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "cpcb-aqi.csv.gz"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 100_000:
        print(f"Dataset already exists: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        print("Delete the file and rerun this script to download again.")
        return

    print("Downloading historical CPCB hourly AQI dataset from GitHub...")
    print("Source: https://github.com/Vonter/india-cpcb-aqi")
    tmp = OUT.with_suffix(OUT.suffix + ".part")
    try:
        with requests.get(URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\rDownloaded {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB", end="")
        # gzip magic bytes: 1f 8b
        with open(tmp, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                raise RuntimeError("Downloaded file is not a gzip file; GitHub may have returned an error page.")
        tmp.replace(OUT)
        print(f"\nSaved: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


if __name__ == "__main__":
    main()
