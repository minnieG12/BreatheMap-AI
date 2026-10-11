# BreatheMap AI

I built BreatheMap AI for the **Air** track of Environmental Hacks. It brings together a three-hour AQI forecast replay, honest model evaluation, and a view of how recently Delhi's monitoring stations reported data.

## The problem I chose

A city-wide AQI number can hide differences between monitoring stations. It also does not tell a resident how recently a station reported an observation or how much trust to place in a short-term prediction. I wanted one simple interface where I could inspect a station, replay a forecast, compare it with the later recorded value, and see monitoring gaps without presenting missing measurements as facts.

## What I built

- **Three-hour forecast replay:** I select a Delhi monitoring station, a historical date, and an hour. The app shows the observed AQI, the ML prediction, the blended estimate, and the recorded target when available.
- **Model evaluation:** I show held-out metrics and compare the blended model with a persistence baseline instead of reporting an accuracy claim without context.
- **AirGap monitoring coverage:** I label stations as Fresh, Stale, or Gap according to the age of their most recent observation at the selected historical timestamp. The source does not include verified station coordinates, so I do not invent AQI values for unmonitored streets.
- **Responsive interface:** I built a green-themed Streamlit UI with a calendar picker, theme-aware charts, high-contrast AQI category labels, and category-based reaction GIFs credited in [`GIF_CREDITS.md`](GIF_CREDITS.md).
- **AWS S3 audit export:** I added a Boto3 path that writes a timestamped run containing the model metrics, the selected forecast replay, and a manifest to a private S3 bucket. Each JSON object is uploaded with SSE-S3 encryption. The app uses the standard AWS credential chain; it never asks me to paste access keys into the UI.

## How the pipeline works

```text
CPCB-derived historical AQI data
        ↓
Download, filter Delhi, reshape to station-hour records
        ↓
Correct duplicated source station IDs and produce a data report
        ↓
Build lag and time features
        ↓
Chronological train / validation / test split
        ↓
HistGradientBoostingRegressor + persistence baseline
        ↓
Validation-selected blend and held-out evaluation
        ↓
Streamlit forecast replay + AirGap coverage view
        ↓
Optional Amazon S3 audit export (metrics, replay, manifest)
```

## Data and attribution

I used historical hourly AQI data from [Vonter/india-cpcb-aqi](https://github.com/Vonter/india-cpcb-aqi), derived from CPCB data. The source dataset is licensed under [ODbL-1.0](https://github.com/Vonter/india-cpcb-aqi/blob/main/LICENSE); its data details are documented in [`DATA.md`](https://github.com/Vonter/india-cpcb-aqi/blob/main/DATA.md).

My prepared dataset contains **294,667 valid station-hour observations across 39 Delhi monitoring-station names**, covering 1 January–31 December 2023. The source reused one `source_station_id` across multiple station names, so I derived a stable station ID from each station name and retained the original ID for traceability. I keep source attribution and licence information with the project.

## Model and measured results

I trained a scikit-learn `HistGradientBoostingRegressor` to predict AQI three hours ahead using historical AQI lags and time features. I compared it with a persistence baseline, using chronological train, validation, and test windows. I selected the blend weight on the validation window and reported final metrics on the held-out test window.

Results from `data/processed/model_metrics.json`:

| Measure | Held-out 2023 result |
|---|---:|
| ML-only MAE | 63.87 AQI points |
| Persistence baseline MAE | 53.39 AQI points |
| Blended forecast MAE | **52.65 AQI points** |
| ML-only RMSE | 81.08 AQI points |
| Persistence baseline RMSE | 79.95 AQI points |
| Blended forecast RMSE | **72.60 AQI points** |
| Blended forecast R² | 0.478 |
| AQI category agreement | 56.0% |

The blend modestly improves MAE and RMSE over the persistence baseline on this held-out period; the standalone ML model does not beat the baseline on MAE. I report this limitation directly rather than claiming high accuracy.

## AWS integration

I integrated **Amazon S3 + Boto3** as a small audit-trail feature. Each upload creates a UTC-timestamped run under `<prefix>/runs/<timestamp>/` containing:

- `model_metrics.json` — the held-out evaluation results;
- `forecast_audit.json` — the selected station, observation and target timestamps, observed/predicted values, and metrics reference;
- `manifest.json` — run metadata, dataset period, station/observation counts, object keys, and AWS identity metadata.

The upload sets `ServerSideEncryption="AES256"`. I keep Block Public Access enabled; the app requires write access to the chosen S3 prefix, not public access. I use an AWS CLI profile or IAM role rather than embedding credentials in the repository.

To configure and verify this integration, follow [`AWS_SETUP.md`](AWS_SETUP.md). The included `scripts/verify_aws_s3.py` uploads a small private health-check object; the final demo should show a successful test and the app's three-object audit upload. Do not claim the S3 path is verified until the upload succeeds in your AWS account.

## Tech stack

- **Python 3.10** — application and pipeline
- **pandas / NumPy** — data preparation and features
- **scikit-learn / joblib** — model training, evaluation, and saved model
- **Streamlit / Plotly / CSS** — interactive dashboard and charts
- **Boto3 / Amazon S3** — timestamped evaluation audit export
- **python-dotenv** — non-secret local configuration

## Run locally

From the repository root on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The repository should include the prepared files needed to run the demo:

- `data/processed/delhi_aqi_long.csv`
- `data/processed/delhi_station_summary.csv`
- `data/processed/data_report.txt`
- `data/processed/model_metrics.json`
- `data/processed/test_predictions.csv`
- `models/aqi_forecast_model.joblib`

To recreate the data/model pipeline from the raw source:

```powershell
python scripts/download_dataset.py
python scripts/prepare_delhi_data.py --start-date 2023-01-01
python scripts/fix_station_ids.py
python scripts/train_aqi_model.py
```

## Limitations and responsible use

- The available records cover **2023 only**. This app is a historical replay, not a live 2026 forecast or an operational warning service.
- The model has imperfect performance and 56.0% AQI category agreement on the reported held-out period.
- The current pipeline uses AQI history and time features; it does not include weather features.
- AirGap reports temporal data coverage. It does not estimate AQI for unmonitored geographic locations because verified station coordinates are not present in the source used here.
- The app's general information is not a substitute for current CPCB/SAFAR advice or medical guidance.
- Reaction GIFs load from Tenor and need an internet connection. Their sources are listed in [`GIF_CREDITS.md`](GIF_CREDITS.md).

## Demo sequence

In my demo, I show the station/date selection, a three-hour historical replay, the observed target comparison, the held-out metrics, the AirGap coverage table, and—after verifying AWS access—the successful S3 audit upload and its objects in the AWS console.

## AI-tool disclosure

I used ChatGPT as an AI coding assistant for project planning, coding guidance, debugging, and documentation. I ran the data preparation and training scripts locally and checked their outputs. I have credited the external dataset and third-party GIFs used in the app; I will disclose any other AI tools if applicable.

## Independence

BreatheMap AI is my independent hackathon prototype. It is not affiliated with CPCB, SAFAR, or any Indian government agency.
