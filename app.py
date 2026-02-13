# app.py
# GenAI Sentiment Analysis Dashboard (continuous -1.0 to +1.0 scoring)
# + graph descriptions + lighter red for Negative

import os
import re
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import openai
from dotenv import load_dotenv

# Load environment variables (.env should include OPENAI_API_KEY=...)
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()


# -----------------------------
# Helpers
# -----------------------------
def get_dataset_path() -> str:
    """
    Assumes:
      /your_repo/
        app.py
        /data/customer_reviews.csv
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return os.path.normpath(csv_path)


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

        # Extract first float-like token defensively
        m = re.search(r"-?\d+(\.\d+)?", raw)
        if not m:
            return 0.0

        score = float(m.group(0))

        # Clamp to [-1.0, 1.0]
        score = max(-1.0, min(1.0, score))

        # Guard against NaN/inf
        if not math.isfinite(score):
            return 0.0

        return score

    except Exception:
        return 0.0


def score_to_label(score: float, pos_threshold: float = 0.2, neg_threshold: float = -0.2) -> str:
    """
    Converts a continuous score into a label for counting/distribution.
    Defaults:
      score >  0.2 => Positive
      score < -0.2 => Negative
      else         => Neutral
    """
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
st.write("This app scores each review’s sentiment on a continuous scale from **-1.0 to +1.0** using OpenAI.")

# Custom colors (you asked: Negative = light red)
COLOR_MAP = {
    "Negative": "#ffb3b3",  # light red
    "Neutral": "#d9d9d9",  # light gray
    "Positive": "#2e7d32",  # green
}

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)

            # Limit rows initially to control API cost
            st.session_state["df"] = df.head(10).copy()

            st.success(f"Dataset loaded successfully! Showing first {len(st.session_state['df'])} rows.")
            st.caption(f"Loaded from: {csv_path}")

        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path in get_dataset_path().")

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

with col2:
    if st.button("🔍 Analyze Sentiment (−1 to +1)"):
        if "df" not in st.session_state:
            st.warning("Please load the dataset first.")
        else:
            df = st.session_state["df"].copy()

            if "SUMMARY" not in df.columns:
                st.error("Your dataset must contain a 'SUMMARY' column.")
            else:
                try:
                    with st.spinner("Analyzing sentiment with OpenAI..."):
                        df["SentimentScore"] = df["SUMMARY"].apply(get_sentiment_score)
                        df["SentimentLabel"] = df["SentimentScore"].apply(score_to_label)
                        st.session_state["df"] = df

                    st.success("Sentiment scoring completed!")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

with col3:
    st.markdown(
        """
        **How scoring works**
        - **-1.0** = very negative sentiment  
        - **0.0** = neutral / mixed sentiment  
        - **+1.0** = very positive sentiment  

        **How labels are assigned (for the “counts” chart)**
        - score < **-0.2** → Negative  
        - -0.2 to 0.2 → Neutral  
        - score > **0.2** → Positive  
        """
    )

# -----------------------------
# Main display
# -----------------------------
if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("🔍 Filter by Product")

    if "PRODUCT" in df.columns:
        products = sorted(df["PRODUCT"].dropna().unique().tolist())
        product = st.selectbox("Choose a product", ["All Products"] + products)

        if product != "All Products":
            filtered_df = df[df["PRODUCT"] == product].copy()
        else:
            filtered_df = df.copy()

        st.subheader(f"📁 Reviews for {product}")
    else:
        product = "All Products"
        filtered_df = df.copy()
        st.info("No PRODUCT column found — showing all rows without product filtering.")

    st.dataframe(filtered_df, use_container_width=True)

    if "SentimentScore" in df.columns and "SentimentLabel" in df.columns:
        st.divider()

        # --- Chart 1: Label distribution (counts) ---
        st.subheader(f"📊 Chart 1: Sentiment Label Distribution — {product}")
        st.caption(
            "What this shows: the **number of reviews** classified as **Negative / Neutral / Positive** "
            "based on the sentiment score thresholds. This helps you see whether the overall feedback mix "
            "is skewed negative or positive."
        )

        sentiment_counts = (
            filtered_df["SentimentLabel"]
            .value_counts()
            .rename_axis("SentimentLabel")
            .reset_index(name="Count")
        )

        order = ["Negative", "Neutral", "Positive"]
        sentiment_counts["SentimentLabel"] = pd.Categorical(
            sentiment_counts["SentimentLabel"], categories=order, ordered=True
        )
        sentiment_counts = sentiment_counts.sort_values("SentimentLabel")

        fig = px.bar(
            sentiment_counts,
            x="SentimentLabel",
            y="Count",
            title=f"Distribution of Sentiment Labels - {product}",
            labels={"SentimentLabel": "Sentiment Category", "Count": "Number of Reviews"},
            color="SentimentLabel",
            color_discrete_map=COLOR_MAP,  # <-- your custom colors
            category_orders={"SentimentLabel": order},
        )
        fig.update_layout(showlegend=False, xaxis_title="Sentiment Category", yaxis_title="Number of Reviews")
        st.plotly_chart(fig, use_container_width=True)

        # --- Chart 2: Score histogram ---
        st.subheader(f"📈 Chart 2: Sentiment Score Histogram — {product}")
        st.caption(
            "What this shows: how sentiment scores are distributed from **-1.0 to +1.0**. "
            "If most bars cluster near **+1**, reviews are strongly positive; near **-1**, strongly negative; "
            "near **0**, mixed/neutral sentiment."
        )

        fig_hist = px.histogram(
            filtered_df,
            x="SentimentScore",
            nbins=20,
            title=f"Sentiment Score Distribution (−1 to +1) - {product}",
            labels={"SentimentScore": "Sentiment Score (−1 to +1)"},
        )
        fig_hist.update_layout(xaxis_title="Sentiment Score (−1 to +1)", yaxis_title="Number of Reviews")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # --- Chart 3: Average score by product ---
        if "PRODUCT" in df.columns:
            st.subheader("📌 Chart 3: Average Sentiment Score by Product")
            st.caption(
                "What this shows: the **average sentiment score** per product. "
                "This is useful for comparing products at a glance. "
                "Closer to **+1** = more positive average sentiment; closer to **-1** = more negative."
            )

            avg_scores = (
                df.groupby("PRODUCT")["SentimentScore"]
                .mean()
                .reset_index()
                .sort_values("SentimentScore")
            )

            fig_avg = px.bar(
                avg_scores,
                x="PRODUCT",
                y="SentimentScore",
                title="Average Sentiment Score by Product (−1 to +1)",
                labels={"PRODUCT": "Product", "SentimentScore": "Avg Sentiment Score"},
            )
            fig_avg.update_layout(xaxis_tickangle=-30, xaxis_title="Product",
                                  yaxis_title="Avg Sentiment Score (−1 to +1)")
            st.plotly_chart(fig_avg, use_container_width=True)

    else:
        st.info("Click **Analyze Sentiment (−1 to +1)** to generate SentimentScore and SentimentLabel.")
