import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pydeck as pdk
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# -----------------------------
# NLTK setup
# -----------------------------
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="🌍 Planet-Level Global Digital Twin", layout="wide")

st.title("🌍 Planet-Level Global Digital Twin")
st.markdown("Real-time data + visualizations + AI summaries")

# -----------------------------
# Example APIs (Free)
# -----------------------------
apis = {
    "Earthquake": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "COVID-19": "https://disease.sh/v3/covid-19/all",
    "News": "https://newsapi.org/v2/top-headlines?country=us&apiKey=demo"  # Replace with your own free key
}

# -----------------------------
# Fetch Earthquake Data
# -----------------------------
st.subheader("🌋 Earthquakes (USGS real-time)")
try:
    eq = requests.get(apis["Earthquake"]).json()
    features = eq["features"]
    eq_data = [{
        "place": f["properties"]["place"],
        "mag": f["properties"]["mag"],
        "lon": f["geometry"]["coordinates"][0],
        "lat": f["geometry"]["coordinates"][1]
    } for f in features if f["properties"]["mag"] is not None]

    df_eq = pd.DataFrame(eq_data)

    st.dataframe(df_eq.head())

    # Pydeck Map
    init = pdk.ViewState(latitude=20, longitude=0, zoom=1)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_eq,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=50000,
        pickable=True
    )

    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=init,
            layers=[layer],
            tooltip={"text": "{place}"}
        )
    )
except Exception as e:
    st.error(f"Earthquake API error: {e}")

# -----------------------------
# Fetch COVID-19 Data
# -----------------------------
st.subheader("🦠 COVID-19 (disease.sh)")
try:
    covid = requests.get(apis["COVID-19"]).json()
    st.metric("Cases", covid["cases"])
    st.metric("Deaths", covid["deaths"])
    st.metric("Recovered", covid["recovered"])
except Exception as e:
    st.error(f"COVID API error: {e}")

# -----------------------------
# Example Visualization
# -----------------------------
st.subheader("📊 Example Chart")
sample = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Value": [10, 23, 45, 12]
})
fig, ax = plt.subplots()
ax.bar(sample["Category"], sample["Value"])
st.pyplot(fig)

# -----------------------------
# AI Summarizers (6 agents)
# -----------------------------
st.subheader("📝 Multi-Agent Summaries")

text = st.text_area("Paste any text (news, report, API data)", 
                    "The planet digital twin is a system that collects real-time data from multiple sources and visualizes it for decision-making.")

if st.button("Generate Summaries"):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    try:
        st.markdown("**1. LexRank Summary**")
        summary = LexRankSummarizer()(parser.document, 2)
        st.write(" ".join(str(s) for s in summary))
    except:
        st.warning("LexRank failed")

    try:
        st.markdown("**2. LSA Summary**")
        summary = LsaSummarizer()(parser.document, 2)
        st.write(" ".join(str(s) for s in summary))
    except:
        st.warning("LSA failed")

    try:
        st.markdown("**3. Luhn Summary**")
        summary = LuhnSummarizer()(parser.document, 2)
        st.write(" ".join(str(s) for s in summary))
    except:
        st.warning("Luhn failed")

    try:
        st.markdown("**4. TextRank Summary**")
        summary = TextRankSummarizer()(parser.document, 2)
        st.write(" ".join(str(s) for s in summary))
    except:
        st.warning("TextRank failed")

    try:
        st.markdown("**5. Lead-3 Summary**")
        st.write(" ".join(text.split(".")[:3]))
    except:
        st.warning("Lead-3 failed")

    try:
        st.markdown("**6. NLTK Frequency Summary**")
        from nltk.corpus import stopwords
        from collections import Counter
        words = [w for w in text.lower().split() if w not in stopwords.words("english")]
        freq = Counter(words).most_common(5)
        st.write("Keywords:", freq)
    except:
        st.warning("NLTK summary failed")
