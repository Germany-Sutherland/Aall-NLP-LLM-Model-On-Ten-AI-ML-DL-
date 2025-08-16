# app.py
# Planet-Level Global Digital Twin — 10-Agent Summarizer (Free-tier Friendly)
# ---------------------------------------------------------------------------
# What this app does
# - Pulls real-time global data from FREE, no-key APIs (USGS, Open‑Meteo, disease.sh, OpenAQ, NASA EONET, GDELT, Wikipedia).
# - Shows a compact dashboard (map + charts + KPIs).
# - Generates **10 different summaries** using lightweight, open-source NLP methods that run on Streamlit Community Cloud.
# - If the tiny Hugging Face model fails to load in free tier, the app **falls back** to extractive methods automatically.
#
# How to deploy (Streamlit Community Cloud)
# 1) Create a public GitHub repo with this `app.py` and a `requirements.txt` (see bottom of this file for contents).
# 2) On streamlit.io → Deploy → point to your repo → select `app.py`.
# 3) No secrets needed.

import os
import sys
import math
import json
import time
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

# --- Lightweight NLP libs ---
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Sumy provides multiple classical summarizers (LexRank, LSA, Luhn, KL, Edmundson)
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Optional abstractive (very small) — with safe fallback
HF_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    HF_AVAILABLE = False

LANG = "english"

st.set_page_config(page_title="🌍 Planet Digital Twin — 10 Agent Summaries", layout="wide", page_icon="🌐")

# Ensure NLTK data is present (quiet download in Cloud)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# -----------------------------
# Sidebar (controls)
# -----------------------------
st.sidebar.header("🌍 Global Digital Twin Controls")
city = st.sidebar.text_input("City (for local context)", "New Delhi")
max_eq_hours = st.sidebar.slider("Earthquakes window (hours)", 6, 48, 24)
show_layers = st.sidebar.multiselect(
    "Data layers",
    ["Earthquakes", "Weather", "Air Quality", "COVID", "Natural Events", "News", "Wikipedia Nearby"],
    default=["Earthquakes", "COVID", "News", "Weather"]
)

st.title("🌍 Planet-Level Global Digital Twin")
st.caption("Free APIs → Visuals → 10-Agent AI/NLP summaries (streamlit free-tier friendly).")

# -----------------------------
# Helpers: API fetchers (no keys)
# -----------------------------
@st.cache_data(show_spinner=False)
def geocode_city(name: str):
    if not name:
        return None
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1, "language": "en", "format": "json"}, timeout=20
    )
    if r.ok and r.json().get("results"):
        res = r.json()["results"][0]
        return {"lat": res["latitude"], "lon": res["longitude"], "name": res["name"], "country": res.get("country")}
    return None

@st.cache_data(show_spinner=False)
def fetch_openmeteo(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m", "timezone": "auto"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_usgs(hours: int = 24):
    starttime = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
        "format": "geojson", "starttime": starttime, "minmagnitude": 2.5
    }, timeout=40)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_covid_global():
    r = requests.get("https://disease.sh/v3/covid-19/all", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_openaq(lat, lon, radius_km=200):
    r = requests.get("https://api.openaq.org/v2/latest", params={
        "coordinates": f"{lat},{lon}", "radius": radius_km*1000, "limit": 100, "order_by": "distance"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_eonet():
    r = requests.get("https://eonet.gsfc.nasa.gov/api/v3/events", params={"status": "open", "limit": 50}, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_gdelt_summary():
    r = requests.get("https://api.gdeltproject.org/api/v2/summary/summary?format=json", timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_wikipedia_nearby(lat, lon):
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
        "gsradius": 15000, "gslimit": 10, "format": "json", "origin": "*"
    }, timeout=20)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Fetch + visualize
# -----------------------------
geo = geocode_city(city)
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

# Weather KPIs
if geo and "Weather" in show_layers:
    try:
        w = fetch_openmeteo(geo["lat"], geo["lon"]) or {}
        cur_temp = (w.get("hourly", {}).get("temperature_2m", [None]) or [None])[0]
        cur_wind = (w.get("hourly", {}).get("wind_speed_10m", [None]) or [None])[0]
        cur_prec = (w.get("hourly", {}).get("precipitation", [None]) or [None])[0]
        with col_k1: st.metric("Temp (°C)", cur_temp)
        with col_k2: st.metric("Wind (m/s)", cur_wind)
        with col_k3: st.metric("Precip (mm)", cur_prec)
        with col_k4: st.metric("City", f"{geo['name']}, {geo.get('country','')}" )
    except Exception as e:
        st.warning(f"Weather unavailable: {e}")
else:
    with col_k1: st.metric("Temp (°C)", "-")
    with col_k2: st.metric("Wind (m/s)", "-")
    with col_k3: st.metric("Precip (mm)", "-")
    with col_k4: st.metric("City", city)

st.divider()

left, right = st.columns([2,1])

with left:
    st.subheader("🌋 Earthquakes (last 6–48 hours)")
    eq_df = pd.DataFrame()
    if "Earthquakes" in show_layers:
        try:
            eq = fetch_usgs(max_eq_hours)
            rows = []
            for f in eq.get("features", []):
                props = f.get("properties", {})
                coords = f.get("geometry", {}).get("coordinates", [None, None])
                if coords and len(coords) >= 2:
                    rows.append({
                        "lon": coords[0], "lat": coords[1],
                        "mag": props.get("mag"),
                        "place": props.get("place"),
                        "time": datetime.utcfromtimestamp((props.get("time") or 0)/1000.0)
                    })
            eq_df = pd.DataFrame(rows)
        except Exception as e:
            st.error(f"USGS error: {e}")

        layers = []
        if not eq_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=eq_df,
                get_position='[lon, lat]',
                get_radius="(mag or 2.5) * 12000",
                radius_min_pixels=2, radius_max_pixels=60, pickable=True
            ))
        init = pdk.ViewState(latitude=0, longitude=0, zoom=1.2)
        st.pydeck_chart(pdk.Deck(initial_view_state=init, map_style=None, layers=layers, tooltip={"text":"{place}
M {mag}"}))

    if geo and "Air Quality" in show_layers:
        st.subheader(f"🌫️ Air Quality near {geo['name']}")
        try:
            aq = fetch_openaq(geo["lat"], geo["lon"], 150)
            stations = []
            for r in aq.get("results", []):
                pm25 = next((m.get("value") for m in r.get("measurements", []) if m.get("parameter")=="pm25"), None)
                if pm25 is not None:
                    stations.append({"station": r.get("location"), "pm25": pm25})
            if stations:
                df_aq = pd.DataFrame(stations).nlargest(10, "pm25")
                st.bar_chart(df_aq.set_index("station")['pm25'])
        except Exception as e:
            st.warning(f"OpenAQ issue: {e}")

with right:
    if "COVID" in show_layers:
        st.subheader("🦠 Global COVID-19 Snapshot")
        try:
            c = fetch_covid_global()
            st.metric("Cases", f"{c.get('cases', 0):,}")
            st.metric("Deaths", f"{c.get('deaths', 0):,}")
            st.metric("Recovered", f"{c.get('recovered', 0):,}")
        except Exception as e:
            st.warning(f"COVID fetch issue: {e}")

    if "News" in show_layers:
        st.subheader("📰 Global News Topics (GDELT)")
        try:
            g = fetch_gdelt_summary()
            terms = [t.get("term") for t in g.get("topterms", {}).get("english", [])][:20]
            if terms:
                df_terms = pd.DataFrame({"topic": terms})
                chart = alt.Chart(df_terms).mark_arc().encode(theta="count()", color="topic", tooltip="topic")
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No topics now.")
        except Exception as e:
            st.warning(f"GDELT issue: {e}")

    if geo and "Wikipedia Nearby" in show_layers:
        st.subheader("📚 Nearby Wikipedia")
        try:
            wq = fetch_wikipedia_nearby(geo["lat"], geo["lon"])
            for it in wq.get("query", {}).get("geosearch", [])[:6]:
                st.markdown(f"- [{it.get('title')}](https://en.wikipedia.org/?curid={it.get('pageid')}) — {it.get('dist')} m")
        except Exception as e:
            st.warning(f"Wikipedia issue: {e}")

st.divider()

# -----------------------------
# Build a single narrative string from the visible data
# -----------------------------
context_bits = []
if 'Earthquakes' in show_layers and 'eq_df' in locals() and not eq_df.empty:
    hotspot = eq_df.nlargest(1, 'mag').iloc[0]
    context_bits.append(
        f"There were {len(eq_df)} earthquakes ≥2.5 in the last {max_eq_hours} hours. Highest magnitude was {hotspot['mag']} near {hotspot['place']}."
    )
if 'COVID' in show_layers and 'c' in locals() and isinstance(c, dict):
    context_bits.append(f"Global COVID snapshot: cases {c.get('cases', 0):,}, deaths {c.get('deaths', 0):,}, recovered {c.get('recovered', 0):,}.")
if geo and 'Air Quality' in show_layers and 'df_aq' in locals():
    worst = df_aq.iloc[0] if not df_aq.empty else None
    if worst is not None:
        context_bits.append(f"Air quality near {geo['name']}: highest PM2.5 ~ {worst['pm25']} µg/m³ at {worst.name}.")
if 'News' in show_layers and 'df_terms' in locals():
    context_bits.append("Top news keywords: " + ", ".join(df_terms['topic'].tolist()[:10]) + ".")

base_text = " ".join(context_bits) if context_bits else "No layers selected or data unavailable."

st.subheader("🤖 10-Agent AI/NLP Summaries")
st.caption("Each card is a different open method. Heavy frameworks are avoided to fit the free tier.")

# -----------------------------
# 10 Summarizers (agents)
# -----------------------------

def abstractive_hf(text: str) -> str:
    """Tiny Hugging Face summarizer with graceful fallback."""
    if not HF_AVAILABLE:
        return "[HF disabled] Using extractive fallback. " + lead3(text)
    try:
        # VERY small models suited for free tier
        sm = os.environ.get("HF_SUMM_MODEL", "sshleifer/distilbart-cnn-12-6")
        pipe = pipeline("summarization", model=sm)
        out = pipe(text[:1500], max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        return out
    except Exception as e:
        return f"[HF fallback] {lead3(text)}"

# Classic extractive helpers

def sumy_summarize(text: str, SummarizerClass, sentences: int = 3) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer(LANG))
    stemmer = Stemmer(LANG)
    summarizer = SummarizerClass(stemmer)
    try:
        summarizer.stop_words = get_stop_words(LANG)
    except Exception:
        pass
    sents = [str(s) for s in summarizer(parser.document, sentences)]
    return " ".join(sents) if sents else lead3(text)


def nltk_freq(text: str, top_n: int = 3) -> str:
    sents = sent_tokenize(text)
    if not sents:
        return text
    stops = set(stopwords.words('english'))
    scores = []
    for s in sents:
        words = [w.lower() for w in word_tokenize(s) if w.isalpha() and w.lower() not in stops]
        scores.append((sum(len(w) for w in words), s))
    top = [s for _, s in sorted(scores, reverse=True)[:top_n]]
    return " ".join(top)


def lead3(text: str) -> str:
    sents = sent_tokenize(text)
    return " ".join(sents[:3]) if sents else text


def keyword_bullets(text: str, k: int = 8) -> str:
    # Simple keyword extraction via frequency
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    stops = set(stopwords.words('english'))
    freq = {}
    for w in words:
        if w in stops: continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]
    return "
".join([f"• {w} ({c})" for w, c in top]) if top else text


def rule_based_alerts(text: str) -> str:
    alerts = []
    # naive thresholds
    if 'earthquakes' in text.lower():
        # try to pull a count
        try:
            count = int([t for t in text.split() if t.isdigit()][0])
            if count >= 50:
                alerts.append("Elevated seismic activity detected globally.")
        except Exception:
            pass
    if 'pm2.5' in text.lower() or 'air quality' in text.lower():
        alerts.append("Check local PM2.5 hotspots for health advisories.")
    if 'covid' in text.lower():
        alerts.append("Monitor COVID trends; maintain public health readiness.")
    if 'news' in text.lower():
        alerts.append("Trending topics may indicate emerging risks or events.")
    return " ".join(alerts) if alerts else "No immediate alerts from simple rules."

# Define 10 agents (names + functions)
agents = [
    ("Abstractive (HF tiny)", lambda t: abstractive_hf(t)),
    ("LexRank (sumy)", lambda t: sumy_summarize(t, LexRankSummarizer, 3)),
    ("LSA (sumy)", lambda t: sumy_summarize(t, LsaSummarizer, 3)),
    ("Luhn (sumy)", lambda t: sumy_summarize(t, LuhnSummarizer, 3)),
    ("KL-Sum (sumy)", lambda t: sumy_summarize(t, KLSummarizer, 3)),
    ("Edmundson (sumy)", lambda t: sumy_summarize(t, EdmundsonSummarizer, 3)),
    ("NLTK Freq", lambda t: nltk_freq(t, 3)),
    ("Lead-3 Baseline", lambda t: lead3(t)),
    ("Keyword Bullets", lambda t: keyword_bullets(t, 8)),
    ("Rule-based Alerts", lambda t: rule_based_alerts(t)),
]

# Render as 2 columns of 5 cards
c1, c2 = st.columns(2)
for i, (name, fn) in enumerate(agents):
    target = c1 if i % 2 == 0 else c2
    with target:
        with st.container(border=True):
            st.markdown(f"**{i+1}. {name}**")
            text = fn(base_text)
            st.write(text if text else base_text)

# -----------------------------
# How to use (inline help)
# -----------------------------
with st.expander("How to use this app"):
    st.markdown(
        """
        **1) Choose your city** in the left sidebar for local context (weather, air quality, nearby Wikipedia).
        
        **2) Pick data layers** (Earthquakes, COVID, Weather, News…). The dashboard updates live from free APIs.
        
        **3) Read the visuals**:
        - **Map** shows earthquakes scaled by magnitude.
        - **Bars/Donut** show air quality and trending news topics.
        - **KPIs** show quick stats (temperature, wind, COVID totals).
        
        **4) Scroll to ‘10-Agent AI/NLP Summaries’**. Each card uses a **different open method** (classical or tiny LLM) to explain what’s happening. If the tiny model can’t load, the app automatically falls back to extractive summaries.
        
        **Tip:** Streamlit free tier has limited RAM. This app avoids heavy models by default. If you want larger LLMs, switch the env var `HF_SUMM_MODEL` or deploy on a beefier machine.
        """
    )

# -----------------------------
# requirements.txt (copy to your repo)
# -----------------------------
if st.checkbox("Show requirements.txt"):
    st.code(
        """
        streamlit
        pandas
        requests
        pydeck
        altair
        nltk
        sumy
        transformers
        torch
        """.strip(), language="text"
    )
