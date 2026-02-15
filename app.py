# app.py
# GenAI Sentiment Analysis Dashboard (continuous -1.0 to +1.0 scoring)
# - Users can upload their own CSV, otherwise app uses local sample dataset
# - Graph descriptions included
# - Negative shown as light red
# - FIXED: session_state resets + DataFrame.update() not adding new columns (charts now render)

import os
import re
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import openai
from dotenv import load_dotenv

# -----------------------------
# OpenAI key + client (robust locally + Streamlit Cloud)
# -----------------------------
api_key = None

# Try Streamlit secrets first (won't crash if secrets missing / misconfigured)
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    api_key = None

# Fall back to .env for local dev
if not api_key:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OpenAI API key not found.\n\n"
        "Local dev:\n"
        "  • Create a `.env` file with `OPENAI_API_KEY=...`\n"
        "  • OR create `.streamlit/secrets.toml` with `OPENAI_API_KEY = \"...\"`\n\n"
        "Streamlit Cloud:\n"
        "  • Set `OPENAI_API_KEY` in App Settings → Secrets."
    )
    st.stop()

client = openai.OpenAI(api_key=api_key)

# -----------------------------
# Helpers
# -----------------------------
def get_dataset_path() -> str:
    """
    Assumes your repo contains:
      /data/customer_reviews.csv
      /app.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return os.path.normpath(csv_path)


def load_local_dataset() -> pd.DataFrame | None:
    try:
        return pd.read_csv(get_dataset_path())
    except Exception as e:
        st.error(f"Could not load local sample dataset: {e}")
        return None


def validate_schema(df: pd.DataFrame) -> None:
    if "SUMMARY" not in df.columns:
        st.error("Dataset must contain a `SUMMARY` column.")
        st.stop()


@st.cache_data
def get_sentiment_score(text: str) -> float:
    """
    Returns a continuous sentiment score in [-1.0, 1.0]
      -1.0 = very negative
       0.0 = neutral
      +1.0 = very positive
    """
    if text is None or (isinstance(text, float) and pd.isna(text)) or str(text).strip() == "":
        return 0.0

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=(
                "You are a sentiment scoring function.\n"
                "Return ONLY a single number between -1.0 and 1.0 (inclusive).\n"
                "-1.0 = very negative, 0.0 = neutral, +1.0 = very positive.\n"
                "No words, no punctuation, no extra text.\n\n"
                f"Review: {text}"
            ),
            temperature=0,
            max_output_tokens=20,
        )

        raw = resp.output[0].content[0].text.strip()
        m = re.search(r"-?\d+(\.\d+)?", raw)
        if not m:
            return 0.0

        score = float(m.group(0))
        score = max(-1.0, min(1.0, score))

        if not math.isfinite(score):
            return 0.0

        return score
    except Exception:
        return 0.0


def score_to_label(score: float, pos_threshold: float = 0.2, neg_threshold: float = -0.2) -> str:
    if score > pos_threshold:
        return "Positive"
    if score < neg_threshold:
        return "Negative"
    return "Neutral"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="GenAI Sentiment Dashboard", layout="wide")

st.title("🔍 GenAI Sentiment Analysis Dashboard")
st.write(
    "This app scores each review’s sentiment on a continuous scale from **-1.0 to +1.0** using OpenAI.\n\n"
    "**If you don’t have your own dataset, the app will use a built-in sample dataset automatically.**"
)

COLOR_MAP = {
    "Negative": "#ffb3b3",  # light red
    "Neutral": "#d9d9d9",   # light gray
    "Positive": "#2e7d32",  # green
}

# -----------------------------
# Data source: upload or local sample
# -----------------------------
st.subheader("📤 Upload Your Review Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV file (required: SUMMARY; recommended: PRODUCT)",
    type=["csv"],
    help="If you do not upload a file, the app will use the built-in sample dataset."
)

# Load data (upload preferred, otherwise local sample)
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        data_source_label = "Uploaded dataset"
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    df_loaded = load_local_dataset()
    data_source_label = "Built-in sample dataset"
    if df_loaded is None:
        st.stop()

validate_schema(df_loaded)

st.caption(
    f"✅ Data source: **{data_source_label}** | Rows: **{len(df_loaded):,}** | Columns: **{len(df_loaded.columns)}**"
)

# IMPORTANT: only initialize df once (or reset only when a new upload happens)
if "df" not in st.session_state:
    st.session_state["df"] = df_loaded.copy()

if uploaded_file is not None:
    # user changed dataset → reset session df
    st.session_state["df"] = df_loaded.copy()

# -----------------------------
# Controls
# -----------------------------
st.subheader("⚙️ Controls")
colA, colB, colC = st.columns([1, 1, 2])

with colA:
    max_to_score = st.number_input(
        "Max reviews to score (controls cost)",
        min_value=1,
        max_value=int(len(st.session_state["df"])),
        value=min(30, int(len(st.session_state["df"]))),
        step=1
    )

# We'll compute filter BEFORE scoring so user can optionally score current selection
df_current = st.session_state["df"]

with colC:
    st.markdown(
        """
        **How scoring works**
        - **-1.0** = very negative sentiment  
        - **0.0** = neutral / mixed sentiment  
        - **+1.0** = very positive sentiment  

        **How labels are assigned**
        - score < **-0.2** → Negative  
        - -0.2 to 0.2 → Neutral  
        - score > **0.2** → Positive  
        """
    )

# -----------------------------
# Filter by product
# -----------------------------
st.subheader("🔍 Filter by Product")

if "PRODUCT" in df_current.columns:
    products = sorted(df_current["PRODUCT"].dropna().unique().tolist())
    product = st.selectbox("Choose a product", ["All Products"] + products)
    if product != "All Products":
        filtered_df = df_current[df_current["PRODUCT"] == product].copy()
    else:
        filtered_df = df_current.copy()
    st.subheader(f"📁 Reviews for {product}")
else:
    product = "All Products"
    filtered_df = df_current.copy()
    st.info("No PRODUCT column found — showing all rows without product filtering.")

# -----------------------------
# Analyze button (FIXED: assigns via .loc so new columns are created)
# -----------------------------
with colB:
    if st.button("🔍 Analyze Sentiment (−1 to +1)"):
        df = st.session_state["df"].copy()

        # Score only the currently filtered rows (up to max_to_score)
        target = filtered_df.head(int(max_to_score)).copy()

        if "SUMMARY" not in target.columns:
            st.error("Dataset must contain a 'SUMMARY' column.")
            st.stop()

        try:
            with st.spinner(f"Scoring sentiment for {len(target):,} reviews..."):
                scores = target["SUMMARY"].apply(get_sentiment_score)
                labels = scores.apply(score_to_label)

                # Write back into the main df using index alignment (CREATES new columns)
                df.loc[target.index, "SentimentScore"] = scores
                df.loc[target.index, "SentimentLabel"] = labels

                st.session_state["df"] = df

            st.success("Sentiment scoring completed!")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Refresh current/filtered after potential scoring
df_current = st.session_state["df"]
if "PRODUCT" in df_current.columns:
    if product != "All Products":
        filtered_df = df_current[df_current["PRODUCT"] == product].copy()
    else:
        filtered_df = df_current.copy()
else:
    filtered_df = df_current.copy()

st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
if "SentimentScore" in df_current.columns and "SentimentLabel" in df_current.columns:
    scored = filtered_df.dropna(subset=["SentimentScore", "SentimentLabel"]).copy()

    if len(scored) == 0:
        st.info("No rows in the current view have been scored yet. Click **Analyze Sentiment**.")
        st.stop()

    st.divider()

    # Chart 1: Breakdown (Neg/Neutral/Pos)
    st.subheader(f"📊 Chart 1: Review Sentiment Breakdown — {product}")
    st.caption(
        "What this shows: the total number of reviews classified as **Negative, Neutral, and Positive** "
        "based on sentiment score thresholds."
    )

    sentiment_counts = (
        scored["SentimentLabel"]
        .value_counts()
        .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
        .reset_index()
    )
    sentiment_counts.columns = ["SentimentLabel", "Count"]

    fig1 = px.bar(
        sentiment_counts,
        x="SentimentLabel",
        y="Count",
        title=f"Review Sentiment Breakdown - {product}",
        labels={"SentimentLabel": "Sentiment Category", "Count": "Number of Reviews"},
        color="SentimentLabel",
        color_discrete_map=COLOR_MAP,
        category_orders={"SentimentLabel": ["Negative", "Neutral", "Positive"]},
    )
    fig1.update_layout(showlegend=False, xaxis_title="Sentiment Category", yaxis_title="Number of Reviews")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Histogram
    st.subheader(f"📈 Chart 2: Sentiment Score Histogram — {product}")
    st.caption(
        "What this shows: how sentiment scores are distributed from **-1.0 to +1.0**. "
        "Clusters near **+1** indicate strongly positive feedback; near **-1** strongly negative; near **0** mixed/neutral."
    )

    fig2 = px.histogram(
        scored,
        x="SentimentScore",
        nbins=20,
        title=f"Sentiment Score Distribution (−1 to +1) - {product}",
        labels={"SentimentScore": "Sentiment Score (−1 to +1)"},
    )
    fig2.update_layout(xaxis_title="Sentiment Score (−1 to +1)", yaxis_title="Number of Reviews")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Average by product
    if "PRODUCT" in df_current.columns:
        st.divider()
        st.subheader("📌 Chart 3: Average Sentiment Score by Product")
        st.caption(
            "What this shows: the **average sentiment score** per product (using only scored rows). "
            "Closer to **+1** = more positive average sentiment; closer to **-1** = more negative."
        )

        scored_all = df_current.dropna(subset=["SentimentScore"]).copy()
        avg_scores = (
            scored_all.groupby("PRODUCT")["SentimentScore"]
            .mean()
            .reset_index()
            .sort_values("SentimentScore")
        )

        fig3 = px.bar(
            avg_scores,
            x="PRODUCT",
            y="SentimentScore",
            title="Average Sentiment Score by Product (−1 to +1)",
            labels={"PRODUCT": "Product", "SentimentScore": "Avg Sentiment Score"},
        )
        fig3.update_layout(xaxis_tickangle=-30, xaxis_title="Product", yaxis_title="Avg Sentiment Score (−1 to +1)")
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Click **Analyze Sentiment (−1 to +1)** to generate SentimentScore and SentimentLabel.")

