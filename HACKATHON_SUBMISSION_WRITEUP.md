# BreatheMap AI — Environmental Hacks submission write-up

I built BreatheMap AI for the **Air** track to make Delhi's air-quality history easier to interpret. Instead of showing a single AQI number, my app lets me select a monitoring station and historical time, replay a three-hour AQI prediction, compare it with the later recorded value, and inspect whether monitoring-station observations are fresh, stale, or missing.

I prepared 294,667 valid station-hour observations across 39 Delhi station names from CPCB-derived historical data covering 2023. I corrected repeated source station IDs by deriving a stable ID from each station name. I trained a scikit-learn HistGradientBoostingRegressor and compared it with a persistence baseline using chronological train, validation, and test windows. On the held-out 2023 test period, the blended forecast produced MAE 52.65 and RMSE 72.60, compared with baseline MAE 53.39 and RMSE 79.95. The standalone model did not beat the baseline on MAE, so I present the result as a modest baseline improvement rather than a high-accuracy live predictor.

I built the interface with Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, and Boto3. I integrated Amazon S3 to preserve a timestamped run audit containing model metrics, the selected forecast replay, and a manifest. The upload uses server-side encryption and relies on a local AWS profile or IAM role; the bucket remains private. **Before submitting this wording, I verify a real S3 upload and show the resulting objects in my demo video.**

The data is historical 2023 data, not a live forecast. AirGap reports temporal monitoring coverage and does not fabricate AQI values for unmonitored streets because the source used here does not provide verified station coordinates.

I used ChatGPT as an AI coding assistant for planning, coding guidance, debugging, and documentation. I credit the CPCB-derived dataset and Tenor GIF sources in the repository.
