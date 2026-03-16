import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix
from databricks import sql
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

# ── Model setup ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model("models/fashion_cnn.keras")

cnn_model = load_cnn_model()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@st.cache_resource
def get_feature_extractor():
    @tf.function
    def extract_features(x):
        for layer in cnn_model.layers:
            x = layer(x)
            if layer.name == "flatten":
                return x
        return x
    dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
    _ = extract_features(dummy)
    return extract_features

extract_features = get_feature_extractor()

@st.cache_resource
def load_embeddings():
    embeddings = np.load("models/catalog_embeddings.npy")
    labels     = np.load("models/catalog_labels.npy")
    return embeddings, labels

cat_embeddings, cat_labels = load_embeddings()

@st.cache_resource
def load_xtest():
    (_, _), (X_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    return X_test

X_test = load_xtest()

def predict_clothing(image_source):
    if isinstance(image_source, str):
        img = Image.open(image_source)
    else:
        img = image_source
    img_gray  = img.convert("L").resize((28, 28))
    img_array = np.array(img_gray)
    if img_array.mean() > 127:
        img_gray  = ImageOps.invert(img_gray)
        img_array = np.array(img_gray)
    arr      = img_array / 255.0
    arr      = arr.reshape(1, 28, 28, 1)
    preds    = cnn_model.predict(arr, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    return {
        "label"      : class_names[top3_idx[0]],
        "confidence" : float(preds[top3_idx[0]]),
        "top3"       : [(class_names[i], float(preds[i])) for i in top3_idx]
    }

def preprocess_image(img):
    img_gray  = img.convert("L").resize((28, 28))
    img_array = np.array(img_gray)
    if img_array.mean() > 127:
        img_gray  = ImageOps.invert(img_gray)
        img_array = np.array(img_gray)
    return (img_array / 255.0).astype(np.float32)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Dashboard",
    page_icon="👗",
    layout="wide"
)

# ── Databricks connection ─────────────────────────────────────────────────────
host      = os.getenv("DATABRICKS_HOST")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token     = os.getenv("DATABRICKS_TOKEN")

@st.cache_data(ttl=300)
def load_predictions():
    with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM workspace.default.fashion_mnist_predictions")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
    return pd.DataFrame(rows, columns=cols)

@st.cache_data(ttl=300)
def load_gold():
    with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM workspace.default.gold_pattern_summary")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
    return pd.DataFrame(rows, columns=cols)

@st.cache_data
def load_trend_data():
    return pd.read_csv("trend_forecast_data.csv")

@st.cache_data
def load_returns_data():
    return pd.read_csv("returns_risk_log.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
DATABRICKS_AVAILABLE = False   # ← set True when credits renew

if DATABRICKS_AVAILABLE:
    with st.spinner("Loading data from Databricks..."):
        pred_df = load_predictions()
        try:
            gold_df  = load_gold()
            has_gold = True
        except:
            has_gold = False
else:
    pred_df  = None
    has_gold = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("👗 Fashion MNIST — ML Pipeline Dashboard")
st.caption("End-to-end pipeline: Kaggle → Databricks Medallion → CNN Model → Predictions")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Pipeline Dashboard",
    "🏷️ Product Tagger",
    "🔍 Visual Search",
    "🏭 Warehouse Scanner",
    "📈 Trend Forecasting",
    "📦 Returns Reduction"
])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
# ═══════════════════════════════════════════════════════════════════════════════
    if not DATABRICKS_AVAILABLE:
        st.warning("⚠️ Databricks credits exhausted — Pipeline Dashboard temporarily unavailable.")
        st.info("Tabs 2–6 are fully functional and run locally.")
    else:
        st.markdown("CNN model performance insights across 10,000 Fashion-MNIST predictions.")

        # ── KPI Metrics ───────────────────────────────────────────────────────
        total    = len(pred_df)
        correct  = pred_df["correct"].sum()
        accuracy = correct / total
        wrong    = total - correct

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", f"{total:,}")
        col2.metric("Correct",           f"{correct:,}")
        col3.metric("Wrong",             f"{wrong:,}")
        col4.metric("Overall Accuracy",  f"{accuracy:.2%}")
        st.divider()

        # ── Row 1: Per-class accuracy + Confusion matrix ──────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Per-class Accuracy")
            class_acc = (
                pred_df.groupby("true_label")["correct"]
                .mean().reset_index()
                .rename(columns={"correct": "accuracy"})
                .sort_values("accuracy", ascending=True)
            )
            fig_bar = px.bar(
                class_acc, x="accuracy", y="true_label",
                orientation="h", color="accuracy",
                color_continuous_scale="teal", range_x=[0, 1],
                labels={"true_label": "", "accuracy": "Accuracy"},
                text=class_acc["accuracy"].apply(lambda x: f"{x:.0%}")
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(coloraxis_showscale=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.subheader("Confusion Matrix")
            labels  = sorted(pred_df["true_label"].unique())
            cm      = confusion_matrix(pred_df["true_label"], pred_df["predicted_label"], labels=labels)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig_cm  = px.imshow(
                cm_norm, x=labels, y=labels,
                color_continuous_scale="Blues",
                labels={"x": "Predicted", "y": "Actual", "color": "Rate"},
                text_auto=".0%"
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()

        # ── Row 2: Prediction distribution + Errors ───────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Prediction Distribution")
            dist = pred_df["predicted_label"].value_counts().reset_index()
            dist.columns = ["label", "count"]
            fig_pie = px.pie(dist, names="label", values="count", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.subheader("Top Misclassifications")
            errors = pred_df[pred_df["correct"] == False]
            mis = (
                errors.groupby(["true_label", "predicted_label"])
                .size().reset_index(name="count")
                .sort_values("count", ascending=False).head(10)
            )
            mis["pair"] = mis["true_label"] + " → " + mis["predicted_label"]
            fig_mis = px.bar(mis, x="count", y="pair", orientation="h",
                             color="count", color_continuous_scale="reds")
            fig_mis.update_layout(coloraxis_showscale=False, height=380,
                                  yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_mis, use_container_width=True)

        st.divider()

        # ── Row 3: Gold table ─────────────────────────────────────────────────
        if has_gold:
            st.subheader("📊 Gold Table — Pattern Summary")
            st.dataframe(gold_df, use_container_width=True)
            st.divider()

        # ── Row 4: Predictions explorer ───────────────────────────────────────
        st.subheader("🔍 Explore Predictions")
        filter_label  = st.selectbox("Filter by true label", ["All"] + sorted(pred_df["true_label"].unique()))
        filter_result = st.radio("Show", ["All", "Correct only", "Wrong only"], horizontal=True)

        filtered = pred_df.copy()
        if filter_label != "All":
            filtered = filtered[filtered["true_label"] == filter_label]
        if filter_result == "Correct only":
            filtered = filtered[filtered["correct"] == True]
        elif filter_result == "Wrong only":
            filtered = filtered[filtered["correct"] == False]

        st.dataframe(filtered.head(200), use_container_width=True)
        st.caption(f"Showing {min(200, len(filtered))} of {len(filtered):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
# ═══════════════════════════════════════════════════════════════════════════════
    st.header("🏷️ Automated Product Tagger")
    st.caption("Upload a clothing image — the model will classify it instantly.")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

    with col_result:
        if uploaded_file:
            result    = predict_clothing(img)
            top_label = result["label"]
            top_conf  = result["confidence"]
            top3      = result["top3"]

            st.markdown("### Prediction")
            if top_conf >= 0.75:
                st.success(f"✅ **{top_label}**")
            elif top_conf >= 0.50:
                st.warning(f"⚠️ **{top_label}** (low confidence)")
            else:
                st.error("❌ Model is uncertain — try a clearer image")

            st.metric("Confidence", f"{top_conf:.1%}")
            st.divider()

            st.markdown("**Top 3 Predictions**")
            top3_labels = [t[0] for t in top3]
            top3_scores = [t[1] for t in top3]

            fig_conf = px.bar(
                x=top3_scores, y=top3_labels,
                orientation="h", color=top3_scores,
                color_continuous_scale="teal", range_x=[0, 1],
                labels={"x": "Confidence", "y": ""},
                text=[f"{s:.1%}" for s in top3_scores]
            )
            fig_conf.update_traces(textposition="outside")
            fig_conf.update_layout(coloraxis_showscale=False, height=200,
                                   margin=dict(l=0, r=20, t=10, b=0))
            st.plotly_chart(fig_conf, use_container_width=True)

            st.divider()
            if st.button("💾 Save to Databricks"):
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS workspace.default.product_tagging_log (
                                filename        STRING,
                                predicted_label STRING,
                                confidence      DOUBLE,
                                tagged_at       TIMESTAMP
                            ) USING DELTA
                        """)
                        cursor.execute(f"""
                            INSERT INTO workspace.default.product_tagging_log VALUES
                            ('{uploaded_file.name}', '{top_label}',
                             {float(top_conf):.4f}, '{now}')
                        """)
                st.success("Saved to Databricks ✅")

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
# ═══════════════════════════════════════════════════════════════════════════════
    st.header("🔍 Visual Search")
    st.caption("Upload an item — find the 5 most visually similar products in the catalog.")

    uploaded_vs = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png"],
        key="visual_search_uploader"
    )

    if uploaded_vs:
        query_img = Image.open(uploaded_vs)
        st.image(query_img, caption="Query Image", width=200)

        arr          = preprocess_image(query_img).reshape(1, 28, 28, 1)
        query_embed  = extract_features(tf.constant(arr)).numpy()
        similarities = cosine_similarity(query_embed, cat_embeddings)[0]
        top_indices  = similarities.argsort()[-5-1:][::-1][:5]

        st.subheader("Top 5 Similar Items")
        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(X_test[idx], width=100, clamp=True)
                st.caption(f"**{class_names[int(cat_labels[idx])]}**\n{similarities[idx]:.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
# ═══════════════════════════════════════════════════════════════════════════════
    st.header("🏭 Warehouse Scanner")
    st.caption("Simulates a real-time conveyor belt — classify items instantly and log to inventory.")

    scan_df       = pd.read_csv("warehouse_scan_log.csv")
    total_scanned = len(scan_df)
    correct       = scan_df["correct"].sum()
    accuracy      = correct / total_scanned
    avg_ms        = scan_df["scan_time_ms"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Items Scanned", f"{total_scanned}")
    col2.metric("Correct",       f"{correct}")
    col3.metric("Accuracy",      f"{accuracy:.0%}")
    col4.metric("Avg Scan Time", f"{avg_ms:.2f} ms")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Scan Results by Category")
        cat_acc = (
            scan_df.groupby("true_label")["correct"]
            .agg(["sum", "count"]).reset_index()
        )
        cat_acc.columns  = ["category", "correct", "total"]
        cat_acc["accuracy"] = (cat_acc["correct"] / cat_acc["total"] * 100).round(1)
        cat_acc = cat_acc.sort_values("accuracy", ascending=True)
        fig = px.bar(cat_acc, x="accuracy", y="category",
                     orientation="h", color="accuracy",
                     color_continuous_scale="teal", range_x=[0, 100],
                     labels={"category": "", "accuracy": "Accuracy %"},
                     text=cat_acc["accuracy"].apply(lambda x: f"{x:.0f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Scan Speed Distribution")
        fig2 = px.histogram(scan_df, x="scan_time_ms", nbins=15,
                            color_discrete_sequence=["#3498db"],
                            labels={"scan_time_ms": "Scan Time (ms)"})
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📋 Raw Scan Log")
    st.dataframe(scan_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
# ═══════════════════════════════════════════════════════════════════════════════
    st.header("📈 Trend Forecasting")
    st.caption("Track weekly category demand to guide buying and merchandising decisions.")

    trend_df    = load_trend_data()
    latest_week = trend_df["week_date"].max()
    latest_df   = trend_df[trend_df["week_date"] == latest_week].copy()
    latest_df   = latest_df.sort_values("wow_change", ascending=False)

    st.subheader(f"Week of {latest_week} — Trend Summary")
    col1, col2, col3 = st.columns(3)

    rising  = latest_df[latest_df["wow_change"] > 0]
    falling = latest_df[latest_df["wow_change"] < 0]

    col1.metric("🔥 Hottest Category",
                rising.iloc[0]["category"] if len(rising) > 0 else "—",
                f"+{rising.iloc[0]['wow_change']:.1f}%" if len(rising) > 0 else "")
    col2.metric("📉 Sharpest Drop",
                falling.iloc[-1]["category"] if len(falling) > 0 else "—",
                f"{falling.iloc[-1]['wow_change']:.1f}%" if len(falling) > 0 else "")
    col3.metric("📅 Weeks Tracked", trend_df["week_date"].nunique())
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Weekly Volume by Category")
        fig = px.line(trend_df, x="week_date", y="count",
                      color="category", markers=True,
                      labels={"week_date": "Week", "count": "Items", "category": "Category"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Week-over-Week Change % (Latest)")
        fig2 = px.bar(latest_df, x="wow_change", y="category",
                      orientation="h",
                      labels={"wow_change": "Change %", "category": ""},
                      color="wow_change",
                      color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                      range_color=[-40, 40])
        fig2.update_layout(coloraxis_showscale=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
# ═══════════════════════════════════════════════════════════════════════════════
    st.header("📦 Returns Reduction")
    st.caption("Flag high-risk items before shipping — reduce returns using model confidence.")

    returns_df = load_returns_data()
    total  = len(returns_df)
    high   = (returns_df["return_risk"] == "High").sum()
    medium = (returns_df["return_risk"] == "Medium").sum()
    low    = (returns_df["return_risk"] == "Low").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items",     f"{total}")
    col2.metric("🚨 High Risk",    f"{high}",   f"{high/total:.0%} of total")
    col3.metric("⚠️ Medium Risk",  f"{medium}", f"{medium/total:.0%} of total")
    col4.metric("✅ Safe to Ship", f"{low}",    f"{low/total:.0%} of total")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Distribution")
        risk_counts = returns_df["return_risk"].value_counts().reindex(["High", "Medium", "Low"])
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
                     color=risk_counts.index,
                     color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Error Rate by Risk Tier")
        error_by_risk = (
            returns_df.groupby("return_risk")
            .agg(error_rate=("correct", lambda x: round((~x).mean() * 100, 1)))
            .reindex(["High", "Medium", "Low"]).reset_index()
        )
        fig2 = px.bar(error_by_risk, x="return_risk", y="error_rate",
                      color="return_risk",
                      color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
                      text=error_by_risk["error_rate"].apply(lambda x: f"{x}%"),
                      labels={"return_risk": "Risk Tier", "error_rate": "Error Rate %"})
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=350, yaxis_range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("🔍 Browse by Risk Tier")
    risk_filter = st.radio("Filter", ["All", "High", "Medium", "Low"], horizontal=True)
    filtered    = returns_df if risk_filter == "All" else returns_df[returns_df["return_risk"] == risk_filter]
    st.dataframe(filtered.head(200), use_container_width=True)
    st.caption(f"Showing {min(200, len(filtered))} of {len(filtered):,} rows")
