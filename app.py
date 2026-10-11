from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "delhi_aqi_long.csv"
MODEL_PATH = ROOT / "models" / "aqi_forecast_model.joblib"
METRICS_PATH = ROOT / "data" / "processed" / "model_metrics.json"
AUDIT_PATH = ROOT / "data" / "processed" / "latest_forecast_audit.json"
HORIZON_HOURS = 3
LAGS = [1, 3, 6, 24]

# Pop-culture reaction GIFs are embedded from Tenor rather than redistributed
# in the repository. Keep source links visible in the UI and in GIF_CREDITS.md.
AQI_STYLES = {
    "Good": {
        "color": "#16794B", "soft": "#DDF7E7", "emoji": "🌿",
        "gif_url": "https://media1.tenor.com/m/_Pm8Z9zVWWkAAAAd/homer-simpson-happy-dance.gif",
        "gif_source": "https://tenor.com/view/homer-simpson-happy-dance-happy-gif-14910605",
        "gif_name": "Homer Simpson happy dance", "mood_line": "Main character energy ✨",
    },
    "Satisfactory": {
        "color": "#4F8A20", "soft": "#EBF8D9", "emoji": "☀️",
        "gif_url": "https://media1.tenor.com/m/4yTYEATOX0gAAAAd/leonardo-dicaprio-cheers.gif",
        "gif_source": "https://tenor.com/view/leonardo-dicaprio-cheers-celebration-gif-13752093",
        "gif_name": "Leonardo DiCaprio cheers", "mood_line": "We love to see it 🥂",
    },
    "Moderate": {
        "color": "#976400", "soft": "#FFF3C4", "emoji": "🌤️",
        "gif_url": "https://media1.tenor.com/m/cJwif62R0wUAAAAd/confused-side-eye.gif",
        "gif_source": "https://tenor.com/view/confused-side-eye-hmm-jim-halpert-the-office-gif-17733506",
        "gif_name": "Jim Halpert side-eye", "mood_line": "Hmm… keep an eye on it 👀",
    },
    "Poor": {
        "color": "#B94B09", "soft": "#FFE4CC", "emoji": "😷",
        "gif_url": "https://media1.tenor.com/m/P_RevRsvfwwAAAAd/spongebob-panic.gif",
        "gif_source": "https://tenor.com/view/spongebob-panic-fire-gif-27173222",
        "gif_name": "SpongeBob panic", "mood_line": "Maybe make it an indoor day 🏠",
    },
    "Very Poor": {
        "color": "#923C83", "soft": "#F7DDF1", "emoji": "🌫️",
        "gif_url": "https://media1.tenor.com/m/rOmVT4_jOUgAAAAd/the-office-panic.gif",
        "gif_source": "https://tenor.com/view/the-office-panic-its-happening-michael-scott-freak-out-gif-5524844",
        "gif_name": "Michael Scott panic", "mood_line": "Okay, let's all stay inside 😬",
    },
    "Severe": {
        "color": "#B42332", "soft": "#FDE0E2", "emoji": "🚨",
        "gif_url": "https://media1.tenor.com/m/T5RIzmp9DXQAAAAd/homer-simpson-hide-in-shrubs.gif",
        "gif_source": "https://tenor.com/view/homer-simpson-hide-in-shrubs-hiding-in-bushes-disappear-into-hedges-embarrassed-gif-5734288276936527220",
        "gif_name": "Homer Simpson disappears into shrubs", "mood_line": "Logging off outside. Bye 👋",
    },
}

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="BreatheMap AI", page_icon="🍃", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
      --bm-forest: #14532d;
      --bm-leaf: #16804a;
      --bm-leaf-bright: #22a05a;
      --bm-light-line: #cfe6d6;
    }
    /* Streamlit theme variables adapt this UI to both Light and Dark modes. */
    .stApp {
      color: var(--text-color);
      background: radial-gradient(ellipse at 8% 0%,
        color-mix(in srgb, var(--background-color) 88%, var(--primary-color) 12%) 0%,
        var(--background-color) 68%);
    }
    [data-testid="stHeader"] {
      background: color-mix(in srgb, var(--background-color) 90%, transparent);
    }
    [data-testid="stMainBlockContainer"] {
      max-width: 1500px;
      padding: 1.2rem clamp(.75rem, 2.6vw, 2.2rem) 2.4rem;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg,
        color-mix(in srgb, var(--secondary-background-color) 88%, var(--primary-color) 12%) 0%,
        var(--secondary-background-color) 100%);
      border-right: 1px solid var(--border-color, rgba(120,150,130,.25));
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
      color: var(--text-color);
    }
    h1, h2, h3, h4 { color: var(--text-color) !important; letter-spacing: -.025em; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text-color); }
    a { color: var(--link-color, var(--primary-color)); }
    [data-testid="stMainBlockContainer"] > div { min-width: 0; }
    [data-testid="stMetric"] {
      background: color-mix(in srgb, var(--secondary-background-color) 96%, var(--primary-color) 4%);
      padding: 15px 16px;
      border: 1px solid var(--border-color, rgba(120,150,130,.25));
      border-radius: 17px;
      box-shadow: 0 5px 18px rgba(20,83,45,.055);
      min-width: 0;
    }
    [data-testid="stMetricLabel"] { color: var(--text-color) !important; opacity: .82; font-weight: 650; }
    [data-testid="stMetricValue"] { color: var(--text-color) !important; font-weight: 850; }
    [data-testid="stMetricDelta"] { font-weight: 650; }
    div[data-testid="stSelectbox"] > label,
    div[data-testid="stDateInput"] > label { font-weight: 750; color: var(--text-color) !important; }
    div[data-testid="stSelectbox"] div[role="combobox"],
    div[data-testid="stDateInput"] input {
      border-radius: 11px;
      color: var(--text-color);
      background: var(--secondary-background-color);
      border-color: var(--border-color, rgba(120,150,130,.3));
    }
    div[data-testid="stSelectbox"] div[role="option"] { color: var(--text-color); }
    div.stButton > button {
      border-radius: 12px; border: 1px solid #138047;
      background: linear-gradient(135deg,#16804a,#1d9a59);
      color: #fff; font-weight: 750;
      box-shadow: 0 5px 14px rgba(22,128,74,.18);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    div.stButton > button:hover {
      border-color: #14532d; background: #14532d; color: #fff;
      transform: translateY(-1px); box-shadow: 0 8px 18px rgba(20,83,45,.24);
    }
    [data-testid="stExpander"] {
      border: 1px solid var(--border-color, rgba(120,150,130,.25));
      border-radius: 14px; background: var(--secondary-background-color);
    }
    [data-testid="stDataFrame"] {
      border: 1px solid var(--border-color, rgba(120,150,130,.25));
      border-radius: 12px; overflow: hidden;
    }
    .hero {
      background: linear-gradient(120deg, #104629 0%, #16804a 56%, #39965d 100%);
      border-radius: 24px; padding: clamp(20px, 3vw, 30px);
      margin: 0 0 18px 0; color: #fff;
      box-shadow: 0 12px 30px rgba(20,83,45,.17);
      position: relative; overflow: hidden;
    }
    .hero:after {
      content: '✦'; position: absolute; right: 6%; top: -20px;
      font-size: 120px; opacity: .10; animation: twinkle 3.3s ease-in-out infinite;
      pointer-events: none;
    }
    .hero-title {
      font-size: clamp(2.05rem, 4.5vw, 3.25rem); font-weight: 900;
      letter-spacing: -.045em; color: #fff !important; line-height: 1.05; margin: 0;
    }
    .hero-subtitle { color: #e2f8e8 !important; font-size: 1rem; margin-top: 10px; margin-bottom: 0; max-width: 780px; line-height: 1.55; }
    .section-kicker { color: var(--primary-color); font-size: .76rem; font-weight: 850; text-transform: uppercase; letter-spacing: .11em; margin-bottom: 2px; }
    .aqi-card {
      border-radius: 20px; padding: 19px 20px;
      background: var(--secondary-background-color);
      color: var(--text-color);
      border: 1px solid var(--border-color, rgba(120,150,130,.27));
      box-shadow: 0 7px 22px rgba(20,83,45,.055); min-height: 166px;
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .aqi-card:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(20,83,45,.10); }
    .aqi-card-label { font-size: .86rem; color: var(--text-color); opacity: .78; font-weight: 750; }
    .aqi-number { font-size: clamp(2.15rem, 4vw, 3.1rem); line-height: 1; font-weight: 900; letter-spacing: -.04em; margin: 10px 0 12px; }
    .aqi-tag {
      display: inline-flex; align-items: center; gap: 7px; border-radius: 999px;
      padding: 7px 13px; font-size: clamp(.95rem, 1.4vw, 1.12rem);
      font-weight: 900; border: 1px solid currentColor;
      animation: tag-pop .42s cubic-bezier(.2,.9,.2,1.1) both;
    }
    .aqi-desc { margin-top: 10px; color: var(--text-color); opacity: .84; font-size: .9rem; line-height: 1.48; }
    .mood-card {
      border-radius: 20px; padding: 13px 16px 11px;
      background: color-mix(in srgb, var(--secondary-background-color) 91%, var(--primary-color) 9%);
      color: var(--text-color); border: 1px solid var(--border-color, rgba(120,150,130,.27));
      min-height: 166px; text-align: center; height: 100%;
    }
    .mood-label { color: var(--text-color); opacity: .8; font-size: .78rem; font-weight: 850; text-transform: uppercase; letter-spacing: .07em; }
    .mood-band { color: var(--text-color); font-weight: 850; padding: 6px 0 3px; }
    .mood-img {
      display: block; margin: 7px auto 2px; width: min(100%, 300px); height: 118px;
      object-fit: contain; border-radius: 12px; animation: meme-float 2.8s ease-in-out infinite;
    }
    .gif-credit { color: var(--text-color) !important; opacity: .72; font-size: .73rem; text-decoration: none; display: inline-block; padding: 3px 0 4px; }
    .gif-credit:hover { opacity: 1; text-decoration: underline; }
    .notice {
      border-left: 5px solid var(--primary-color);
      background: color-mix(in srgb, var(--secondary-background-color) 86%, var(--primary-color) 14%);
      color: var(--text-color); padding: 13px 16px; border-radius: 10px;
      margin: 12px 0 19px; line-height: 1.55;
    }
    .small-note { color: var(--text-color); opacity: .72; font-size: .83rem; }
    .footer-note { padding-top: 12px; margin-top: 22px; border-top: 1px solid var(--border-color, rgba(120,150,130,.25)); color: var(--text-color); opacity: .72; font-size: .78rem; line-height: 1.5; }

    .fine-print { display: block; opacity: .72; font-size: .76rem; margin-top: 4px; }
    .status-note { animation: card-in .35s ease-out both; }
    .coverage-summary {
      border-radius: 16px; padding: 18px 19px; min-height: 112px;
      border: 1px solid var(--border-color, rgba(120,150,130,.27));
      background: var(--secondary-background-color); color: var(--text-color);
      transition: transform .2s ease, box-shadow .2s ease;
    }
    .coverage-summary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(20,83,45,.09); }
    .coverage-summary.fresh { border-left: 5px solid #16A34A; }
    .coverage-summary.stale { border-left: 5px solid #D97706; }
    .coverage-summary.gap { border-left: 5px solid #DC2626; }
    .coverage-summary-title { font-size: 1.12rem; font-weight: 850; color: var(--text-color); }
    .coverage-summary-copy { margin-top: 5px; line-height: 1.45; color: var(--text-color); }
    .coverage-summary-foot { margin-top: 9px; font-size: .74rem; opacity: .68; color: var(--text-color); }
    .airgap-mood { border: 1px solid var(--border-color, rgba(120,150,130,.27)); background: var(--secondary-background-color); border-radius: 16px; padding: 10px; text-align: center; height: 100%; }
    .airgap-mood img { width: min(100%, 210px); height: 85px; object-fit: contain; border-radius: 10px; animation: meme-float 2.8s ease-in-out infinite; }
    .airgap-mood a { display: block; color: var(--text-color); font-size: .75rem; opacity: .76; margin-top: 3px; }
    [data-testid="stPlotlyChart"] { border: 1px solid #294637; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 24px rgba(12,35,22,.13); }
    [data-testid="stDataFrame"] { max-width: 100%; }
    .mood-card { overflow: hidden; }
    .mood-band { line-height: 1.35; }
    @keyframes card-in { from { opacity: .4; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    [data-testid="stAlert"] { border-radius: 13px; }
    @keyframes meme-float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
    @keyframes twinkle { 0%,100% { transform: rotate(0deg) scale(1); opacity: .08; } 50% { transform: rotate(12deg) scale(1.08); opacity: .18; } }
    @keyframes tag-pop { 0% { transform: scale(.92); opacity: .4; } 100% { transform: scale(1); opacity: 1; } }
    @media (max-width: 780px) {
      [data-testid="stMainBlockContainer"] { padding: .8rem .65rem 1.5rem; }
      .hero { padding: 20px 18px; border-radius: 18px; }
      .aqi-card { padding: 15px; min-height: 0; }
      .mood-card { min-height: 0; }
      .mood-img { height: 100px; }
      [data-testid="stMetric"] { padding: 10px; border-radius: 13px; }
    }
    @media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def aqi_band(value: float) -> tuple[str, str]:
    if value <= 50:
        return "Good", "Looks good. Enjoy your day."
    if value <= 100:
        return "Satisfactory", "Mostly okay. Sensitive groups: stay aware."
    if value <= 200:
        return "Moderate", "Sensitive people may feel discomfort outdoors."
    if value <= 300:
        return "Poor", "Limit long outdoor workouts."
    if value <= 400:
        return "Very Poor", "Cut down time outdoors."
    return "Severe", "Keep outdoor exposure to a minimum."


@st.cache_data(show_spinner="Loading historical Delhi observations…")
def load_data(path_string: str) -> pd.DataFrame:
    frame = pd.read_csv(path_string, parse_dates=["timestamp"])
    required = {"timestamp", "station_id", "station_name", "aqi"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset is missing columns: {sorted(required - set(frame.columns))}")
    frame["station_id"] = frame["station_id"].astype(str)
    frame["station_name"] = frame["station_name"].astype(str)
    frame["aqi"] = pd.to_numeric(frame["aqi"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "station_id", "station_name", "aqi"])
    frame = frame[(frame["aqi"] >= 0) & (frame["aqi"] <= 500)]
    frame = frame.drop_duplicates(["station_id", "timestamp"], keep="last")
    return frame.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading trained AQI model…")
def load_model(path_string: str):
    return joblib.load(path_string)


def build_one_row(station_frame: pd.DataFrame, selected_ts: pd.Timestamp, all_station_ids: list[str]):
    series = station_frame.set_index("timestamp")["aqi"]
    if selected_ts not in series.index:
        return None, "There is no measured AQI at that station and hour."
    values = {"aqi_now": float(series.loc[selected_ts])}
    for lag in LAGS:
        lag_ts = selected_ts - pd.Timedelta(hours=lag)
        if lag_ts not in series.index:
            return None, f"Missing exact {lag}-hour lag at {lag_ts}. Choose another hour with complete history."
        values[f"aqi_lag_{lag}h"] = float(series.loc[lag_ts])
    values["hour"] = selected_ts.hour
    values["day_of_week"] = selected_ts.dayofweek
    values["month"] = selected_ts.month
    values["hour_sin"] = np.sin(2 * np.pi * selected_ts.hour / 24)
    values["hour_cos"] = np.cos(2 * np.pi * selected_ts.hour / 24)
    values["doy_sin"] = np.sin(2 * np.pi * selected_ts.dayofyear / 365.25)
    values["doy_cos"] = np.cos(2 * np.pi * selected_ts.dayofyear / 365.25)
    values["station_id"] = station_frame["station_id"].iloc[0]
    row = pd.DataFrame([values])
    row["station_id"] = pd.Categorical(row["station_id"], categories=all_station_ids)
    return row, None


def render_aqi_card(label: str, value: float, band: str, description: str) -> str:
    style = AQI_STYLES[band]
    return f"""
    <div class="aqi-card">
      <div class="aqi-card-label">{label}</div>
      <div class="aqi-number" style="color:{style['color']}">{value:.0f}</div>
      <div><span class="aqi-tag" style="color:{style['color']};background:{style['soft']}">{style['emoji']} {band.upper()}</span></div>
      <div class="aqi-desc">{description}</div>
    </div>
    """


def main():
    st.markdown(
        """
        <div class="hero">
          <h1 class="hero-title">BreatheMap AI</h1>
          <p class="hero-subtitle">Explore the air. Preview the next 3 hours. Spot monitoring gaps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><b>Historical replay · 2023 data.</b> Not live conditions or official health advice.</div>',
        unsafe_allow_html=True,
    )

    missing = [label for label, path in [("processed AQI data", DATA_PATH), ("trained model", MODEL_PATH), ("model metrics", METRICS_PATH)] if not path.exists()]
    if missing:
        st.error("Missing required files: " + ", ".join(missing))
        st.code("python scripts/fix_station_ids.py\npython scripts/train_aqi_model.py")
        st.stop()

    try:
        df = load_data(str(DATA_PATH))
        bundle = load_model(str(MODEL_PATH))
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    all_station_ids = sorted(df["station_id"].unique().tolist())
    station_name_by_id = df.drop_duplicates("station_id").set_index("station_id")["station_name"].to_dict()
    station_counts = df.groupby("station_id").size().sort_values(ascending=False)
    default_station = str(station_counts.index[0])

    st.sidebar.markdown("## 🍃 Explore the air")
    st.sidebar.caption("Pick a station, date and hour.")
    selected_station = st.sidebar.selectbox(
        "Monitoring station",
        options=all_station_ids,
        index=all_station_ids.index(default_station),
        format_func=lambda sid: station_name_by_id.get(sid, sid),
    )
    station_frame = df[df["station_id"] == selected_station].sort_values("timestamp").copy()
    times = set(pd.Timestamp(t) for t in station_frame["timestamp"].tolist())
    valid_times = [t for t in sorted(times) if all((t - pd.Timedelta(hours=lag)) in times for lag in LAGS)]
    if not valid_times:
        st.error("This station has no timestamp with all required hourly lags. Select another station.")
        st.stop()
    valid_with_target = [t for t in valid_times if t + pd.Timedelta(hours=HORIZON_HOURS) in times]
    candidates = valid_with_target if valid_with_target else valid_times
    date_options = sorted({t.date() for t in candidates})
    default_date = date_options[-1]
    chosen_date = st.sidebar.date_input(
        "Historical date",
        value=default_date,
        min_value=date_options[0],
        max_value=date_options[-1],
        format="DD/MM/YYYY",
        help="Choose a date from the calendar. Only the available historical period is supported.",
    )
    same_day = [t for t in candidates if t.date() == chosen_date]
    if not same_day:
        nearest = min(date_options, key=lambda d: abs((d - chosen_date).days))
        st.sidebar.warning(f"No complete replay hours on {chosen_date:%d %b %Y}. Using nearest available date: {nearest:%d %b %Y}.")
        chosen_date = nearest
        same_day = [t for t in candidates if t.date() == chosen_date]
    selected_ts = st.sidebar.selectbox(
        "Observation hour",
        options=same_day,
        index=len(same_day) - 1,
        format_func=lambda t: t.strftime("%H:%M"),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("🕰️ Historical replay · not live")
    st.sidebar.caption("Only times with enough history are enabled.")

    feature_row, problem = build_one_row(station_frame, pd.Timestamp(selected_ts), all_station_ids)
    if problem:
        st.error(problem)
        st.stop()
    feature_cols = bundle["feature_columns"]
    feature_row = feature_row[feature_cols]
    ml_prediction = float(np.clip(bundle["model"].predict(feature_row)[0], 0, 500))
    current_aqi = float(feature_row["aqi_now"].iloc[0])
    weight = float(bundle.get("model_weight", 0.0))
    ensemble_prediction = float(np.clip((1.0 - weight) * current_aqi + weight * ml_prediction, 0, 500))
    target_ts = pd.Timestamp(selected_ts) + pd.Timedelta(hours=HORIZON_HOURS)
    actual_series = station_frame.set_index("timestamp")["aqi"]
    actual_target = float(actual_series.loc[target_ts]) if target_ts in actual_series.index else None
    current_band, current_advice = aqi_band(current_aqi)
    predicted_band, predicted_advice = aqi_band(ensemble_prediction)

    st.markdown(f"<div class='section-kicker'>YOUR STATION</div><h2 style='margin-top:0'>{station_name_by_id.get(selected_station, selected_station)}</h2>", unsafe_allow_html=True)
    st.caption(f"{pd.Timestamp(selected_ts):%d %b %Y · %H:%M}  →  {target_ts:%H:%M} · 3-hour replay")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observed now", f"{current_aqi:.0f}")
    m2.metric("ML estimate · +3h", f"{ml_prediction:.0f}")
    m3.metric("Combined estimate", f"{ensemble_prediction:.0f}")
    if actual_target is not None:
        m4.metric("Recorded · +3h", f"{actual_target:.0f}", f"Error {abs(actual_target - ensemble_prediction):.1f} pts", delta_color="off")
    else:
        m4.metric("Recorded · +3h", "—", "No reading at target hour", delta_color="off")

    st.markdown("### 🌿 Air snapshot")
    c1, c2, c3 = st.columns([1, 1, 1.05], gap="medium")
    with c1:
        st.markdown(render_aqi_card("Observed at selected hour", current_aqi, current_band, current_advice), unsafe_allow_html=True)
    with c2:
        st.markdown(render_aqi_card("Predicted in 3 hours", ensemble_prediction, predicted_band, predicted_advice), unsafe_allow_html=True)
    with c3:
        style = AQI_STYLES[predicted_band]
        mood_art = (
            f'<a href="{style["gif_source"]}" target="_blank" rel="noopener noreferrer" '
            f'aria-label="Open source for {style["gif_name"]}">'
            f'<img class="mood-img" src="{style["gif_url"]}" '
            f'alt="{style["gif_name"]} reaction GIF for {predicted_band} air quality" '
            f'loading="lazy" referrerpolicy="no-referrer" />'
            f'</a>'
        )
        mood_html = (
            f'<div class="mood-card"><div class="mood-label">MOOD · {predicted_band.upper()}</div>'
            f'{mood_art}'
            f'<div class="mood-band">{style["mood_line"]}</div>'
            f'<a class="gif-credit" href="{style["gif_source"]}" target="_blank" rel="noopener noreferrer">GIF via Tenor ↗</a>'
            f'</div>'
        )
        st.markdown(mood_html, unsafe_allow_html=True)

    if predicted_band in {"Poor", "Very Poor", "Severe"}:
        st.markdown(f"<div class='notice status-note'><b>{AQI_STYLES[predicted_band]['emoji']} {predicted_band}:</b> {predicted_advice} <span class='fine-print'>Historical estimate · check current official guidance.</span></div>", unsafe_allow_html=True)

    chart_frame = station_frame[(station_frame["timestamp"] <= pd.Timestamp(selected_ts)) & (station_frame["timestamp"] >= pd.Timestamp(selected_ts) - pd.Timedelta(hours=48))].tail(80)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_frame["timestamp"], y=chart_frame["aqi"], mode="lines", name="Observed AQI", line={"color": "#42D483", "width": 3.2}, hovertemplate="%{x}<br>Observed AQI: %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[target_ts], y=[ensemble_prediction], mode="markers", name="Estimate (+3h)", marker={"size": 15, "symbol": "diamond", "color": AQI_STYLES[predicted_band]["color"], "line": {"color": "#F0F7F2", "width": 2}}, hovertemplate="Forecast target: %{x}<br>Predicted AQI: %{y:.0f}<extra></extra>"))
    if actual_target is not None:
        fig.add_trace(go.Scatter(x=[target_ts], y=[actual_target], mode="markers", name="Recorded (+3h)", marker={"size": 12, "symbol": "circle-open", "color": "#A7F3D0", "line": {"width": 3}}, hovertemplate="Recorded target: %{x}<br>Actual AQI: %{y:.0f}<extra></extra>"))
    # Keep the chart on a deliberately dark, high-contrast canvas so axis labels
    # remain readable even when Streamlit's theme is switched from its menu.
    plot_text = "#F0F7F2"
    plot_muted = "#B7CFC0"
    plot_grid = "#315341"
    chart_bg = "#0E2117"
    fig.update_layout(
        template="plotly_dark",
        title={"text": "AQI · Last 48 hours + 3-hour estimate", "font": {"size": 19, "color": plot_text}},
        xaxis_title="Time", yaxis_title="AQI", legend_title="", height=390,
        margin={"l": 18, "r": 18, "t": 58, "b": 20},
        paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
        font={"color": plot_text, "size": 12}, hovermode="x unified",
        legend={"font": {"color": plot_text}, "bgcolor": "rgba(14,33,23,0.75)"},
    )
    fig.update_xaxes(gridcolor=plot_grid, zerolinecolor=plot_grid, tickfont={"color": plot_muted}, title_font={"color": plot_text}, linecolor=plot_grid, showline=True)
    fig.update_yaxes(range=[0, 500], gridcolor=plot_grid, zerolinecolor=plot_grid, tickfont={"color": plot_muted}, title_font={"color": plot_text}, linecolor=plot_grid, showline=True)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})

    with st.expander("📈 Model evaluation — held-out 2023 test period", expanded=True):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ensemble MAE", f"{metrics['blended_ensemble_MAE']:.2f}")
        e2.metric("Persistence MAE", f"{metrics['persistence_baseline_MAE']:.2f}")
        e3.metric("Ensemble RMSE", f"{metrics['blended_ensemble_RMSE']:.2f}")
        e4.metric("AQI band agreement", f"{metrics['aqi_category_agreement']:.1%}")
        st.caption(f"ML-only MAE: {metrics['ml_model_MAE']:.2f} · Persistence MAE: {metrics['persistence_baseline_MAE']:.2f} · ML weight: {metrics['validation_selected_ml_model_weight']:.0%}.")
        st.caption("Held-out historical 2023 test period; not validated for current conditions.")

    st.markdown("---")
    st.markdown("### 📡 AirGap · Coverage check")
    st.caption("Are the monitors reporting at this hour? Gaps are flagged—not guessed.")
    coverage_rows = []
    for sid, sub in df.groupby("station_id", sort=False):
        prior = sub[sub["timestamp"] <= pd.Timestamp(selected_ts)]
        if prior.empty:
            coverage_rows.append({"Station": station_name_by_id.get(sid, sid), "Last seen": "No reading", "Age (h)": None, "Last AQI": None, "Status": "No data"})
            continue
        last = prior.iloc[-1]
        age = (pd.Timestamp(selected_ts) - last["timestamp"]).total_seconds() / 3600
        status = "Fresh" if age <= 3 else ("Stale" if age <= 12 else "Gap")
        coverage_rows.append({"Station": station_name_by_id.get(sid, sid), "Last seen": last["timestamp"].strftime("%d %b · %H:%M"), "Age (h)": round(age, 1), "Last AQI": int(round(float(last["aqi"]))), "Status": status})
    coverage = pd.DataFrame(coverage_rows)
    status_order = {"Gap": 0, "No data": 0, "Stale": 1, "Fresh": 2}
    coverage["_priority"] = coverage["Status"].map(status_order)
    coverage = coverage.sort_values(["_priority", "Age (h)"], ascending=[True, False], na_position="first").drop(columns="_priority")
    fresh = int((coverage["Status"] == "Fresh").sum())
    stale = int((coverage["Status"] == "Stale").sum())
    gaps = int(coverage["Status"].isin(["Gap", "No data"]).sum())
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Stations", len(coverage))
    g2.metric("Fresh · ≤3h", fresh)
    g3.metric("Stale · 4–12h", stale)
    g4.metric("Gaps · >12h", gaps)

    if gaps == 0 and stale == 0:
        coverage_mood = {"gif_url": AQI_STYLES["Good"]["gif_url"], "gif_source": AQI_STYLES["Good"]["gif_source"], "gif_name": "Homer Simpson happy dance", "heading": "All monitors accounted for", "line": "All stations have a recent reading at this replay time."}
        coverage_tone = "fresh"
    elif gaps <= 3:
        coverage_mood = {"gif_url": AQI_STYLES["Moderate"]["gif_url"], "gif_source": AQI_STYLES["Moderate"]["gif_source"], "gif_name": "Jim Halpert side-eye", "heading": "A few blind spots", "line": f"{gaps} station(s) have a long gap. Treat local coverage with care."}
        coverage_tone = "stale"
    else:
        coverage_mood = {"gif_url": AQI_STYLES["Very Poor"]["gif_url"], "gif_source": AQI_STYLES["Very Poor"]["gif_source"], "gif_name": "Michael Scott panic", "heading": "Coverage needs attention", "line": f"{gaps} stations have missing or old readings at this time."}
        coverage_tone = "gap"
    airgap_left, airgap_right = st.columns([2.2, 1], gap="medium")
    with airgap_left:
        box_class = "coverage-summary " + coverage_tone
        st.markdown(f"<div class='{box_class}'><div class='coverage-summary-title'>{coverage_mood['heading']}</div><div class='coverage-summary-copy'>{coverage_mood['line']}</div><div class='coverage-summary-foot'>Coverage is based on the selected historical timestamp.</div></div>", unsafe_allow_html=True)
    with airgap_right:
        st.markdown(f"<div class='airgap-mood'><img src='{coverage_mood['gif_url']}' alt='{coverage_mood['gif_name']}' loading='lazy' /><a href='{coverage_mood['gif_source']}' target='_blank' rel='noopener noreferrer'>Coverage mood ↗</a></div>", unsafe_allow_html=True)

    def coverage_color(value: str) -> str:
        if value == "Fresh": return "background-color: #DCFCE7; color: #166534; font-weight: 750"
        if value == "Stale": return "background-color: #FEF3C7; color: #92400E; font-weight: 750"
        return "background-color: #FEE2E2; color: #991B1B; font-weight: 750"
    styled_coverage = coverage.style.map(coverage_color, subset=["Status"]).format({"Age (h)": lambda x: "—" if pd.isna(x) else f"{x:.1f}", "Last AQI": lambda x: "—" if pd.isna(x) else f"{int(x)}"})
    st.dataframe(styled_coverage, width="stretch", hide_index=True, height=300)
    st.caption("No verified coordinates in this source yet, so AirGap reports monitoring coverage—not AQI for unmeasured streets.")

    st.markdown("---")
    st.markdown("### ☁️ AWS · S3 audit trail")
    st.caption("Save the evaluation metrics, selected forecast replay, and a run manifest to a private S3 bucket. Credentials are read from your local AWS profile or role; never paste keys here.")
    default_bucket = os.getenv("AWS_S3_BUCKET", "")
    default_prefix = os.getenv("AWS_S3_PREFIX", "breathemap-ai")
    default_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", ""))
    default_profile = os.getenv("AWS_PROFILE", "")
    with st.expander("AWS destination", expanded=not bool(default_bucket)):
        bucket = st.text_input("S3 bucket name", value=default_bucket, placeholder="your-globally-unique-bucket")
        aws_prefix = st.text_input("Object prefix", value=default_prefix, help="Objects will be stored below this prefix; keep the bucket private.")
        aws_region = st.text_input("AWS region (optional if your profile has one)", value=default_region, placeholder="ap-south-1")
        aws_profile = st.text_input("AWS CLI profile (optional)", value=default_profile, placeholder="default or breathemap-ai")
        st.caption("Recommended: keep S3 Block Public Access enabled. This app writes JSON audit objects only and applies SSE-S3 encryption on upload.")
    if st.button("Upload evaluation + replay audit to S3", type="primary", width="stretch"):
        if not bucket.strip():
            st.error("Enter your S3 bucket name first.")
        elif not METRICS_PATH.exists():
            st.error("Model metrics are missing. Run scripts/train_aqi_model.py first.")
        else:
            try:
                import boto3
                from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound

                session = boto3.Session(
                    profile_name=aws_profile.strip() or None,
                    region_name=aws_region.strip() or None,
                )
                identity = session.client("sts").get_caller_identity()
                s3 = session.client("s3", region_name=aws_region.strip() or session.region_name or None)
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                clean_prefix = aws_prefix.strip().strip("/") or "breathemap-ai"
                run_prefix = f"{clean_prefix}/runs/{run_id}"

                metrics_payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                audit = {
                    "project": "BreatheMap AI",
                    "schema_version": "1.0",
                    "mode": "historical replay, not live data",
                    "station": station_name_by_id.get(selected_station, selected_station),
                    "observation_timestamp": str(selected_ts),
                    "forecast_target_timestamp": str(target_ts),
                    "forecast_horizon_hours": HORIZON_HOURS,
                    "observed_aqi": float(current_aqi),
                    "ml_prediction_aqi": float(ml_prediction),
                    "blend_weight": float(weight),
                    "blended_prediction_aqi": float(ensemble_prediction),
                    "recorded_target_aqi": None if actual_target is None or pd.isna(actual_target) else float(actual_target),
                    "model_test_metrics": metrics_payload,
                }
                AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
                AUDIT_PATH.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
                manifest = {
                    "project": "BreatheMap AI",
                    "run_id_utc": run_id,
                    "aws_account_id": identity.get("Account"),
                    "aws_caller_arn": identity.get("Arn"),
                    "dataset_period": "2023-01-01 through 2023-12-31",
                    "dataset_mode": "historical; not live air-quality data",
                    "station_count": int(df["station_id"].nunique()),
                    "source_observations": int(len(df)),
                    "forecast_horizon_hours": HORIZON_HOURS,
                    "objects": {
                        "metrics": f"{run_prefix}/model_metrics.json",
                        "forecast_audit": f"{run_prefix}/forecast_audit.json",
                        "manifest": f"{run_prefix}/manifest.json",
                    },
                    "encryption": "SSE-S3 (AES256)",
                }
                objects = [
                    (manifest["objects"]["metrics"], metrics_payload),
                    (manifest["objects"]["forecast_audit"], audit),
                    (manifest["objects"]["manifest"], manifest),
                ]
                for key, payload in objects:
                    s3.put_object(
                        Bucket=bucket.strip(),
                        Key=key,
                        Body=json.dumps(payload, indent=2, allow_nan=False).encode("utf-8"),
                        ContentType="application/json",
                        ServerSideEncryption="AES256",
                    )
                st.success(f"AWS upload successful · account ending {str(identity.get('Account', ''))[-4:]} · {len(objects)} JSON objects saved.")
                st.code("\n".join(f"s3://{bucket.strip()}/{key}" for key, _ in objects), language="text")
                st.caption("The run is timestamped for auditability. Keep the bucket private; show this success state and the matching objects in the AWS console during the demo.")
            except Exception as exc:
                # Show useful, non-secret error text without dumping credentials/environment variables.
                from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
                if isinstance(exc, NoCredentialsError):
                    st.error("AWS credentials were not found. Configure an AWS CLI profile or role, then restart Streamlit from that same terminal.")
                elif isinstance(exc, ProfileNotFound):
                    st.error("That AWS profile was not found. Check the profile name or leave it blank to use the default credential chain.")
                elif isinstance(exc, ClientError):
                    code = exc.response.get("Error", {}).get("Code", "AWS service error")
                    st.error(f"AWS rejected the operation ({code}). Check the bucket name, bucket region, and s3:PutObject permission for the selected prefix.")
                else:
                    st.error(f"S3 upload failed: {type(exc).__name__}: {exc}")
                st.caption("Do not paste access keys into this app or commit them to GitHub. Use a local profile or IAM role.")

    st.markdown("<div class='footer-note'>Historical CPCB-derived dataset · ODbL-1.0 attribution: Vonter/india-cpcb-aqi. BreatheMap AI is an independent prototype, not an official CPCB/SAFAR service.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
