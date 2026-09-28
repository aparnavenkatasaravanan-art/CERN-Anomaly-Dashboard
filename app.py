import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="CERN Anomaly Dashboard",
    layout="wide"
)
# ----------------------------
# Dashboard Styling
# ----------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stMetric"] {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
}

h1, h2, h3 {
    font-weight: 600;
}

[data-testid="stSidebar"] {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("cern_anomaly_results.csv")

# ----------------------------
# Title
# ----------------------------
st.markdown(
    """
    <div style="
        padding: 25px 30px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 25px;
    ">
        <h1 style="margin-bottom: 5px;">
            🧪 CERN Electron Collision Anomaly Detection
        </h1>
        <p style="
            font-size: 17px;
            margin-top: 5px;
        ">
            AI-powered analysis and anomaly detection of electron collision events
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")

# ----------------------------
# Anomaly Label Preparation
# ----------------------------

if "anomaly_label" in df.columns:

    # Isolation Forest:
    #  1  = Normal
    # -1  = Anomaly

    df["status"] = df["anomaly_label"].map({
        1: "Normal",
        -1: "Anomaly"
    })

# ----------------------------
# Sidebar Filters
# ----------------------------

st.sidebar.markdown("## 🔬 Dashboard Controls")

st.sidebar.markdown(
    "Use the controls below to explore collision events."
)

st.sidebar.divider()

st.sidebar.markdown("### Event Classification")

anomaly_filter = st.sidebar.selectbox(
    "Select Event Type",
    ["All", "Normal", "Anomaly"]
)

st.sidebar.divider()

st.sidebar.markdown("### 📌 Detection Method")

st.sidebar.info(
    "Isolation Forest is used to identify unusual "
    "electron collision patterns."
)

if anomaly_filter == "Normal":
    df = df[df["status"] == "Normal"]

elif anomaly_filter == "Anomaly":
    df = df[df["status"] == "Anomaly"]

# ----------------------------
# KPI Cards
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

total_events = len(df)

if "anomaly_label" in df.columns:
    anomaly_events = len(df[df["anomaly_label"] == -1])
    normal_events = len(df[df["anomaly_label"] == 1])
else:
    anomaly_events = 0
    normal_events = total_events

anomaly_rate = (anomaly_events / total_events) * 100 if total_events > 0 else 0

with col1:
    st.markdown("### 📊 Total Events")
    st.metric(
        label="",
        value=f"{total_events:,}"
    )

with col2:
    st.markdown("### ✅ Normal Events")
    st.metric(
        label="",
        value=f"{normal_events:,}"
    )

with col3:
    st.markdown("### 🚨 Anomalous Events")
    st.metric(
        label="",
        value=f"{anomaly_events:,}"
    )

with col4:
    st.markdown("### 📈 Anomaly Rate")
    st.metric(
        label="",
        value=f"{anomaly_rate:.2f}%"
    )
st.divider()

# ----------------------------
# Electron Energy Analysis
# ----------------------------

st.divider()

st.subheader("⚡ Electron Energy Analysis")

if "E1" in df.columns and "E2" in df.columns:

    energy_sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    energy_fig = px.scatter(
        energy_sample,
        x="E1",
        y="E2",
        color="status",
        hover_data=[
            col for col in [
                "Event",
                "E1",
                "E2",
                "pt1",
                "pt2",
                "anomaly_score"
            ]
            if col in energy_sample.columns
        ],
        title="E1 vs E2 Energy Distribution",
        labels={
            "E1": "Electron 1 Energy",
            "E2": "Electron 2 Energy",
            "status": "Event Type"
        }
    )

    energy_fig.update_layout(
        height=500,
        hovermode="closest",
        legend_title="Classification",
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=50
        )
    )

    st.plotly_chart(
        energy_fig,
        use_container_width=True,
        key="energy_scatter"
    )

    st.caption(
        "This plot compares the energy of the two electrons "
        "and highlights differences between normal and anomalous events."
    )

else:

    st.warning(
        "Electron energy data (E1/E2) is not available."
    )
# ----------------------------
# Invariant Mass Analysis
# ----------------------------

st.divider()

st.subheader("⚛️ Invariant Mass Analysis")

if "M" in df.columns:

    mass_fig = px.histogram(
        df,
        x="M",
        nbins=60,
        color="status",
        barmode="overlay",
        opacity=0.75,
        title="Invariant Mass Distribution by Event Classification",
        labels={
            "M": "Invariant Mass",
            "count": "Number of Events",
            "status": "Event Type"
        }
    )

    mass_fig.update_layout(
        height=500,
        hovermode="x unified",
        legend_title="Classification",
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=50
        )
    )

    st.plotly_chart(
        mass_fig,
        use_container_width=True,
        key="invariant_mass_distribution"
    )

    st.caption(
        "The invariant mass distribution helps compare collision "
        "patterns between normal and anomalous events."
    )

else:

    st.warning(
        "Invariant mass data is not available in the dataset."
    )
# ----------------------------
# Top Anomalies
# ----------------------------
st.subheader("Top Anomalous Events")

if "anomaly_score" in df.columns:

    top = df.sort_values(
        "anomaly_score",
        ascending=False
    ).head(10)

    columns_needed = []

    for col in ["Event", "E1", "E2", "pt1", "pt2", "M", "anomaly_score"]:
        if col in top.columns:
            columns_needed.append(col)

    st.dataframe(top[columns_needed])

# ----------------------------
# Anomaly Score Analysis
# ----------------------------

st.divider()

st.subheader("🎯 Anomaly Score Analysis")

if "anomaly_score" in df.columns:

    score_fig = px.histogram(
        df,
        x="anomaly_score",
        nbins=60,
        color="status",
        barmode="overlay",
        opacity=0.75,
        title="Distribution of Anomaly Scores",
        labels={
            "anomaly_score": "Anomaly Score",
            "count": "Number of Events",
            "status": "Event Type"
        },
        hover_data=[
            col for col in ["Event", "E1", "E2"]
            if col in df.columns
        ]
    )

    score_fig.update_layout(
        height=500,
        hovermode="x unified",
        legend_title="Classification",
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=50
        )
    )

    st.plotly_chart(
        score_fig,
        use_container_width=True,
        key="anomaly_score_distribution"
    )

    st.caption(
        "The distribution shows how anomaly scores vary across "
        "normal and anomalous collision events."
    )

else:

    st.warning(
        "Anomaly score data is not available in the dataset."
    )

# ----------------------------
# AI Feature Importance
# ----------------------------
st.divider()
st.subheader("🧠 AI Feature Importance")
st.caption("Feature importance estimated from the anomaly detection model.")

try:
    shap_df = pd.read_csv("shap_feature_importance.csv")
    shap_df.columns = shap_df.columns.str.strip()

    if "Feature" in shap_df.columns and "Importance" in shap_df.columns:

        shap_df["Importance"] = pd.to_numeric(
            shap_df["Importance"],
            errors="coerce"
        )

        shap_df = shap_df.dropna(
            subset=["Feature", "Importance"]
        )

        shap_df = shap_df.sort_values(
            "Importance",
            ascending=True
        )

        shap_fig = px.bar(
            shap_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Contribution to Anomaly Detection",
            labels={
                "Importance": "Importance",
                "Feature": "Collision Feature"
            },
            text="Importance"
        )

        shap_fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        shap_fig.update_layout(
            height=450,
            margin=dict(l=40, r=40, t=70, b=40),
            xaxis_title="Importance",
            yaxis_title="Feature"
        )

        st.plotly_chart(
            shap_fig,
            use_container_width=True,
            key="shap_feature_importance"
        )

        st.info(
            "Higher importance indicates that the feature contributed more "
            "to distinguishing unusual collision events."
        )

    else:
        st.error(
            "SHAP file must contain 'Feature' and 'Importance' columns."
        )

except FileNotFoundError:
    st.error(
        "shap_feature_importance.csv was not found in the project folder."
    )

# ----------------------------
# Top 10 Anomalous Events
# ----------------------------

st.divider()

st.subheader("🚨 Top 10 Most Anomalous Events")

if "anomaly_label" in df.columns and "anomaly_score" in df.columns:

    anomaly_data = df[
        df["anomaly_label"] == -1
    ].copy()

    anomaly_data = anomaly_data.sort_values(
        "anomaly_score",
        ascending=False
    ).head(10)

    display_columns = [
        "Event",
        "E1",
        "E2",
        "pt1",
        "pt2",
        "M",
        "anomaly_score"
    ]

    available_columns = [
        col for col in display_columns
        if col in anomaly_data.columns
    ]

    display_table = anomaly_data[
        available_columns
    ].copy()

    if "anomaly_score" in display_table.columns:
        display_table["anomaly_score"] = (
            display_table["anomaly_score"].round(4)
        )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Events are ranked by anomaly score, with higher-scoring "
        "events shown first."
    )
# ----------------------------
# Event Explorer
# ----------------------------

st.divider()

st.subheader("🔎 Event Explorer")

if "Event" in df.columns:

    event_list = df["Event"].astype(str).tolist()

    selected_event = st.selectbox(
        "Select an Event ID",
        event_list,
        key="event_explorer"
    )

    selected_row = df[
        df["Event"].astype(str) == selected_event
    ]

    if not selected_row.empty:

        event = selected_row.iloc[0]

        # Event classification
        if event["status"] == "Anomaly":
            st.warning(
                f"🚨 Event {selected_event} is classified as ANOMALY"
            )
        else:
            st.success(
                f"✅ Event {selected_event} is classified as NORMAL"
            )

        # Key measurements
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Energy E1",
                f"{event['E1']:.3f}"
            )

        with col2:
            st.metric(
                "Energy E2",
                f"{event['E2']:.3f}"
            )

        with col3:
            st.metric(
                "Momentum pt1",
                f"{event['pt1']:.3f}"
            )

        with col4:
            st.metric(
                "Invariant Mass",
                f"{event['M']:.3f}"
            )

        # Detailed information
        st.markdown("### 📋 Event Details")

        details = pd.DataFrame({
            "Parameter": [
                "Event ID",
                "E1",
                "E2",
                "pt1",
                "pt2",
                "Invariant Mass",
                "Anomaly Score",
                "Classification"
            ],
            "Value": [
                event["Event"],
                event["E1"],
                event["E2"],
                event["pt1"],
                event["pt2"],
                event["M"],
                event["anomaly_score"],
                event["status"]
            ]
        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

else:

    st.warning(
        "Event ID data is not available."
    )
# ----------------------------
# Anomaly Explanation
# ----------------------------

st.divider()
st.subheader("🧠 Why Is This Event Anomalous?")

anomaly_events = df[df["anomaly_label"] == -1].copy()

if not anomaly_events.empty:

    anomaly_event_ids = anomaly_events["Event"].astype(str).tolist()

    selected_anomaly = st.selectbox(
        "Select an Anomalous Event",
        anomaly_event_ids,
        key="why_anomalous_event"
    )

    event_data = anomaly_events[
        anomaly_events["Event"].astype(str) == selected_anomaly
    ].iloc[0]

    st.warning(
        f"🚨 Event {selected_anomaly} is classified as ANOMALY"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Energy E1", f"{event_data['E1']:.3f}")

    with col2:
        st.metric("Energy E2", f"{event_data['E2']:.3f}")

    with col3:
        st.metric("Momentum pt1", f"{event_data['pt1']:.3f}")

    with col4:
        st.metric("Invariant Mass", f"{event_data['M']:.3f}")

    st.markdown("### 🎯 Anomaly Score")

    score = float(event_data["anomaly_score"])

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Anomaly Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 0.3], "name": "Low"},
                    {"range": [0.3, 0.6], "name": "Medium"},
                    {"range": [0.6, 1], "name": "High"}
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": score
                }
            }
        )
    )

    gauge_fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=60, b=20)
    )

    st.plotly_chart(
        gauge_fig,
        use_container_width=True,
        key="anomaly_score_gauge"
    )

    st.markdown("### Feature Values")

    feature_values = pd.DataFrame({
        "Feature": ["E1", "E2", "pt1", "pt2", "M"],
        "Value": [
            event_data["E1"],
            event_data["E2"],
            event_data["pt1"],
            event_data["pt2"],
            event_data["M"]
        ]
    })

    st.dataframe(
        feature_values,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This event was identified as anomalous because its combined "
        "collision-feature pattern differs from the dominant patterns "
        "observed in the dataset."
    )

    st.markdown("### 📊 Comparison with Normal Events")

    comparison_features = ["E1", "E2", "pt1", "pt2", "M"]

    normal_data = df[df["anomaly_label"] == 1]

    comparison_data = []

    for feature in comparison_features:

        if feature in df.columns:

            anomaly_value = pd.to_numeric(
                event_data[feature],
                errors="coerce"
            )

            normal_values = pd.to_numeric(
                normal_data[feature],
                errors="coerce"
            ).dropna()

            if not normal_values.empty:

                comparison_data.append({
                    "Feature": feature,
                    "Selected Anomaly": anomaly_value,
                    "Normal Mean": normal_values.mean()
                })

    comparison_df = pd.DataFrame(comparison_data)

    if not comparison_df.empty:

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        comparison_long = comparison_df.melt(
            id_vars="Feature",
            value_vars=[
                "Selected Anomaly",
                "Normal Mean"
            ],
            var_name="Type",
            value_name="Value"
        )

        comparison_fig = px.bar(
            comparison_long,
            x="Feature",
            y="Value",
            color="Type",
            barmode="group",
            title="Selected Anomaly vs Normal Mean",
            labels={
                "Value": "Feature Value",
                "Feature": "Collision Feature",
                "Type": "Comparison"
            }
        )

        comparison_fig.update_layout(
            height=450,
            legend_title="Comparison",
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=50
            )
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
            key="anomaly_normal_comparison"
        )

    else:
        st.info(
            "Comparison data is not available for this event."
        )

else:

    st.info(
        "No anomalous events are available in the current selection."
    )
 # ---------------------------------------------------------
 # Anomaly Score Gauge
 # ---------------------------------------------------------

st.markdown("### 🎯 Anomaly Score")

score = float(event_data["anomaly_score"])

gauge_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=score,
        title={
            "text": "Anomaly Score"
        },
        gauge={
            "axis": {
                "range": [0, 1]
            },
            "bar": {
                "thickness": 0.25
            },
            "steps": [
                {
                    "range": [0, 0.3],
                    "name": "Low"
                },
                {
                    "range": [0.3, 0.6],
                    "name": "Medium"
                },
                {
                    "range": [0.6, 1],
                    "name": "High"
                }
            ],
            "threshold": {
                "line": {
                    "width": 4
                },
                "thickness": 0.75,
                "value": score
            }
        }
    )
)

gauge_fig.update_layout(
    height=350,
    margin=dict(
        l=40,
        r=40,
        t=60,
        b=20
    )
)

st.plotly_chart(
    gauge_fig,
    use_container_width=True,
    key="anomaly_score_gauge"
)
# ----------------------------
# Project Information
# ----------------------------

st.divider()

st.subheader("📌 About This Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔬 Project")
    st.write(
        "CERN Electron Collision Anomaly Detection"
    )

    st.markdown("### 🤖 Detection Method")
    st.write(
        "Isolation Forest based anomaly detection"
    )

with col2:
    st.markdown("### 📊 Dataset")
    st.write(
        f"{len(df):,} collision events analyzed"
    )

    st.markdown("### 🎯 Purpose")
    st.write(
        "Identify unusual patterns in electron collision "
        "events using machine learning."
    )

st.caption(
    "CERN Electron Collision Anomaly Detection Dashboard"
)
        