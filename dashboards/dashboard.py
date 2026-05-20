import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from confluent_kafka import Consumer
import streamlit.components.v1 as components
import os
import sys

# Adjust Python Path to resolve local imports cleanly from workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_REPORT_PATH = os.path.abspath(os.path.join(BASE_DIR, "../ml/models_registry/evaluation_report.json"))
from dashboards.drift_report import build_drift_report_html

st.set_page_config(page_title="Enterprise Financial Crime Monitoring", layout="wide", page_icon="🏦")

if 'transactions' not in st.session_state:
    st.session_state.transactions = []

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Initialize Kafka Consumer for Dashboard
if 'consumer' not in st.session_state:
    conf = {
        'bootstrap.servers': os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
        'group.id': 'streamlit_dashboard_group',
        'auto.offset.reset': 'latest',
        'max.poll.interval.ms': 3600000  # Prevent timeout if browser tab sleeps
    }
    st.session_state.consumer = Consumer(conf)
    try:
        st.session_state.consumer.subscribe(['fraud-alerts'])
    except Exception as e:
        st.error(f"Failed to connect to Kafka: {e}")

st.title("🏦 Enterprise Financial Crime Monitoring")
st.markdown("""
**Distributed Architecture:** Kafka (Stream) → Redis (Feature Store) → Neo4j (Graph) → PyTorch LSTM (DL) → XGBoost/IsoForest (ML) → Evidently AI (MLOps Drift).
""")


def load_evaluation_report():
    if not os.path.exists(EVALUATION_REPORT_PATH):
        return None
    try:
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_model_card(model_name, metrics):
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("f1")
    roc_auc = metrics.get("roc_auc")

    st.markdown(f"#### {model_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}" if precision is not None else "N/A")
    c2.metric("Recall", f"{recall:.3f}" if recall is not None else "N/A")
    c3.metric("F1", f"{f1:.3f}" if f1 is not None else "N/A")
    c4.metric("ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")

    cm = metrics.get("confusion_matrix", {})
    matrix = [
        [cm.get("true_negative", 0), cm.get("false_positive", 0)],
        [cm.get("false_negative", 0), cm.get("true_positive", 0)],
    ]
    fig = px.imshow(
        matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legit", "Fraud"],
        y=["Legit", "Fraud"],
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


LIVE_TABLE_FORMAT = {
    "amount": "${:,.2f}",
    "supervised_risk": "{:.3f}",
    "unsupervised_risk": "{:.3f}",
    "sequence_risk": "{:.3f}",
    "combined_risk": "{:.3f}",
    "processing_ms": "{:.1f}",
    "total_amount": "${:,.2f}",
    "max_risk": "{:.3f}",
}


def existing_columns(df, columns):
    return [col for col in columns if col in df.columns]


def highlight_fraud_rows(row):
    if row.get("is_fraud") or row.get("action") == "BLOCK":
        return ["background-color: rgba(255, 50, 50, 0.25)"] * len(row)
    return [""] * len(row)


def render_live_table(title, df, columns, empty_message, rows=15, highlight=False):
    st.markdown(f"#### {title}")
    if df.empty:
        st.info(empty_message)
        return

    visible_cols = existing_columns(df, columns)
    if not visible_cols:
        st.info("No matching columns available yet.")
        return

    view = df[visible_cols].head(rows)
    fmt = {col: LIVE_TABLE_FORMAT[col] for col in visible_cols if col in LIVE_TABLE_FORMAT}
    styled = view.style.format(fmt)
    if highlight:
        styled = styled.apply(highlight_fraud_rows, axis=1)

    st.dataframe(styled, use_container_width=True, height=min(420, 110 + (len(view) * 35)))


def build_blocked_accounts(df):
    if df.empty or "nameOrig" not in df.columns:
        return pd.DataFrame()

    is_fraud_mask = df["is_fraud"].astype(bool) if "is_fraud" in df.columns else pd.Series(False, index=df.index)
    action_mask = df["action"] == "BLOCK" if "action" in df.columns else pd.Series(False, index=df.index)
    blocked_df = df[is_fraud_mask | action_mask].copy()
    if blocked_df.empty:
        return pd.DataFrame()

    agg = blocked_df.groupby("nameOrig").agg(
        blocked_transactions=("nameOrig", "count"),
        total_amount=("amount", "sum"),
        max_risk=("combined_risk", "max"),
        latest_reason=("reason", "first"),
    ).reset_index()
    return agg.sort_values(["blocked_transactions", "max_risk"], ascending=[False, False])


def model_signal_table(df, risk_col, threshold=0.6):
    if df.empty or risk_col not in df.columns:
        return pd.DataFrame()
    return df[df[risk_col] >= threshold].sort_values(risk_col, ascending=False)


def review_queue_table(df):
    if df.empty or "combined_risk" not in df.columns:
        return pd.DataFrame()
    if "action" in df.columns:
        allowed_df = df[df["action"] != "BLOCK"].copy()
    else:
        allowed_df = df.copy()

    velocity_mask = allowed_df["velocity_5m"] >= 4 if "velocity_5m" in allowed_df.columns else pd.Series(False, index=allowed_df.index)
    return allowed_df[
        (allowed_df["combined_risk"] >= 0.35) | velocity_mask
    ].sort_values("combined_risk", ascending=False)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Stream", "📈 Model Performance",
    "🕸️ Network Graph", "📊 MLOps Drift Monitor"
])

# ─────────────────────────────────────────────
# TAB 1: LIVE STREAM
# ─────────────────────────────────────────────
@st.fragment(run_every=1)
def live_dashboard():
    if st.session_state.is_running:
        try:
            msgs = st.session_state.consumer.consume(num_messages=10, timeout=0.5)
            for msg in msgs:
                if msg.error() is None:
                    tx_data = json.loads(msg.value().decode('utf-8'))
                    st.session_state.transactions.insert(0, tx_data)

            if len(st.session_state.transactions) > 100:
                st.session_state.transactions = st.session_state.transactions[:100]

        except Exception as e:
            st.error(f"Kafka Error: {e}")
            st.session_state.is_running = False

    col1, col2 = st.columns([1, 4])

    with col1:
        st.subheader("System Controls")
        start_btn = st.button("▶️ Start Stream", disabled=st.session_state.is_running)
        stop_btn = st.button("⏹️ Pause Stream", disabled=not st.session_state.is_running)

        if start_btn:
            st.session_state.is_running = True
            st.rerun()
        if stop_btn:
            st.session_state.is_running = False
            st.rerun()

        st.markdown("---")
        st.metric(label="Transactions Analyzed", value=len(st.session_state.transactions))

        fraud_count = sum(1 for t in st.session_state.transactions if t.get('is_fraud'))
        st.metric(label="Blocked (High Risk)", value=fraud_count, delta=fraud_count, delta_color="inverse")

        # Average latency metric
        if st.session_state.transactions:
            latencies = [t.get('processing_ms', 0) for t in st.session_state.transactions if t.get('processing_ms')]
            if latencies:
                st.metric(label="Avg Latency", value=f"{np.mean(latencies):.1f} ms")

    with col2:
        st.subheader("Live Transaction Operations")

        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)

            is_fraud_mask = df["is_fraud"].astype(bool) if "is_fraud" in df.columns else pd.Series(False, index=df.index)
            action_mask = df["action"] == "BLOCK" if "action" in df.columns else pd.Series(False, index=df.index)
            blocked_df = df[is_fraud_mask | action_mask].copy()
            blocked_accounts_df = build_blocked_accounts(df)
            xgb_df = model_signal_table(df, "supervised_risk")
            iso_df = model_signal_table(df, "unsupervised_risk")
            lstm_df = model_signal_table(df, "sequence_risk")
            review_df = review_queue_table(df)

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Incoming", len(df))
            k2.metric("Blocked", len(blocked_df))
            k3.metric("Review", len(review_df))
            k4.metric("XGBoost Flags", len(xgb_df))
            k5.metric("IsoForest Flags", len(iso_df))
            k6.metric("LSTM Flags", len(lstm_df))

            if "prediction_outcome" in df.columns:
                outcome_counts = df["prediction_outcome"].value_counts()
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("True Positive", int(outcome_counts.get("TRUE_POSITIVE", 0)))
                a2.metric("False Positive", int(outcome_counts.get("FALSE_POSITIVE", 0)))
                a3.metric("False Negative", int(outcome_counts.get("FALSE_NEGATIVE", 0)))
                a4.metric("True Negative", int(outcome_counts.get("TRUE_NEGATIVE", 0)))

            table_tabs = st.tabs(
                [
                    "Incoming",
                    "Blocked",
                    "Blocked Accounts",
                    "XGBoost",
                    "Isolation Forest",
                    "LSTM Sequence",
                    "Review Queue",
                ]
            )

            ledger_cols = [
                "timestamp", "event_source", "step", "nameOrig", "nameDest", "type", "amount", "velocity_5m",
                "supervised_risk", "unsupervised_risk", "sequence_risk",
                "combined_risk", "ground_truth_is_fraud", "prediction_outcome", "reason", "action"
            ]
            alert_cols = [
                "timestamp", "event_source", "step", "nameOrig", "nameDest", "type", "amount",
                "velocity_5m", "combined_risk", "ground_truth_is_fraud", "prediction_outcome", "reason", "action"
            ]
            model_cols = [
                "timestamp", "event_source", "step", "nameOrig", "nameDest", "type", "amount",
                "supervised_risk", "unsupervised_risk", "sequence_risk",
                "combined_risk", "ground_truth_is_fraud", "prediction_outcome", "reason", "action"
            ]
            account_cols = [
                "nameOrig", "blocked_transactions", "total_amount", "max_risk", "latest_reason"
            ]

            with table_tabs[0]:
                render_live_table(
                    "Incoming Transactions",
                    pd.DataFrame(st.session_state.transactions),
                    ledger_cols,
                    "No incoming transactions yet.",
                    rows=20,
                    highlight=True,
                )

            with table_tabs[1]:
                render_live_table(
                    "Blocked Transactions",
                    blocked_df,
                    alert_cols,
                    "No blocked transactions yet.",
                    rows=20,
                    highlight=True,
                )

            with table_tabs[2]:
                render_live_table(
                    "Blocked Accounts",
                    blocked_accounts_df,
                    account_cols,
                    "No blocked accounts yet.",
                    rows=20,
                )

            with table_tabs[3]:
                render_live_table(
                    "XGBoost High-Risk Signals",
                    xgb_df,
                    model_cols,
                    "No XGBoost risk scores above 0.600 yet.",
                    rows=20,
                    highlight=True,
                )

            with table_tabs[4]:
                render_live_table(
                    "Isolation Forest Anomaly Signals",
                    iso_df,
                    model_cols,
                    "No Isolation Forest risk scores above 0.600 yet.",
                    rows=20,
                    highlight=True,
                )

            with table_tabs[5]:
                render_live_table(
                    "LSTM Sequence Signals",
                    lstm_df,
                    model_cols,
                    "No LSTM sequence risk scores above 0.600 yet.",
                    rows=20,
                    highlight=True,
                )

            with table_tabs[6]:
                render_live_table(
                    "Manual Review Queue",
                    review_df,
                    ledger_cols,
                    "No allowed transactions currently need review.",
                    rows=20,
                    highlight=True,
                )

            if len(df) > 1 and 'action' in df.columns:
                st.markdown("#### Risk Timeline")
                df['color'] = df['action'].map({'BLOCK': '#ff2b2b', 'ALLOW': '#0068c9'}).fillna('#0068c9')
                df['tx_index'] = range(len(df))
                st.scatter_chart(df, x='tx_index', y='combined_risk', color='color')

            if len(df) > 3 and 'amount' in df.columns:
                st.markdown("#### Transaction Amount Distribution")
                fig = px.histogram(df, x='amount', nbins=30, color_discrete_sequence=['#0068c9'])
                fig.update_layout(height=250, margin=dict(t=10, b=30, l=40, r=10))
                st.plotly_chart(fig, use_container_width=True)

with tab1:
    live_dashboard()

# ─────────────────────────────────────────────
# TAB 2: MODEL PERFORMANCE
# ─────────────────────────────────────────────
with tab2:
    st.subheader("📈 Model Performance & Contribution Analysis")
    evaluation_report = load_evaluation_report()

    if evaluation_report:
        st.markdown("### Offline Training Evaluation")
        meta1, meta2, meta3 = st.columns(3)
        meta1.metric("Evaluation Rows", evaluation_report.get("row_count", 0))
        meta2.metric("Fraud Cases", evaluation_report.get("fraud_count", 0))
        meta3.metric("Fraud Rate", f"{evaluation_report.get('fraud_rate', 0.0) * 100:.2f}%")
        st.caption(
            f"Dataset: {evaluation_report.get('dataset', 'unknown')} | "
            f"Generated: {evaluation_report.get('generated_at', 'unknown')}"
        )

        model_metrics = evaluation_report.get("models", {})
        metric_cols = st.columns(2)
        with metric_cols[0]:
            if "xgboost" in model_metrics:
                render_model_card("XGBoost Supervised Classifier", model_metrics["xgboost"])
        with metric_cols[1]:
            if "isolation_forest" in model_metrics:
                render_model_card("Isolation Forest Anomaly Detector", model_metrics["isolation_forest"])
    else:
        st.info("No offline evaluation report found yet. Run `python ml/evaluate_model.py` or retrain with `python ml/train_model.py`.")

    st.markdown("---")
    st.markdown("### Live Stream Behavior")

    if len(st.session_state.transactions) > 5:
        df = pd.DataFrame(st.session_state.transactions)

        # Top KPIs
        k1, k2, k3, k4 = st.columns(4)
        fraud_count = sum(1 for t in st.session_state.transactions if t.get('is_fraud'))
        total = len(st.session_state.transactions)

        with k1:
            st.metric("Total Analyzed", total)
        with k2:
            st.metric("Fraud Blocked", fraud_count)
        with k3:
            st.metric("Block Rate", f"{(fraud_count/max(1,total)*100):.1f}%")
        with k4:
            latencies = [t.get('processing_ms', 0) for t in st.session_state.transactions if t.get('processing_ms')]
            st.metric("Avg Latency", f"{np.mean(latencies):.1f} ms" if latencies else "N/A")

        st.markdown("---")

        # Fraud reason breakdown
        if 'reason' in df.columns:
            st.markdown("### Detection Reason Breakdown")
            reason_counts = df[df['reason'] != 'Normal']['reason'].value_counts()
            if not reason_counts.empty:
                fig = px.pie(
                    values=reason_counts.values, names=reason_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fraud detections yet.")

        # Per-model risk distributions
        risk_cols = ['supervised_risk', 'unsupervised_risk', 'sequence_risk']
        available_risk_cols = [c for c in risk_cols if c in df.columns]

        if available_risk_cols:
            st.markdown("### Per-Model Risk Score Distributions")
            colors = {
                'supervised_risk': '#0068c9',
                'unsupervised_risk': '#ff8c00',
                'sequence_risk': '#9b59b6'
            }
            labels = {
                'supervised_risk': 'XGBoost (Supervised)',
                'unsupervised_risk': 'Isolation Forest (Unsupervised)',
                'sequence_risk': 'LSTM (Deep Learning)'
            }

            cols = st.columns(len(available_risk_cols))
            for col, risk_col in zip(cols, available_risk_cols):
                with col:
                    fig = px.histogram(
                        df, x=risk_col, nbins=20,
                        color_discrete_sequence=[colors.get(risk_col, '#0068c9')]
                    )
                    fig.update_layout(
                        title=labels.get(risk_col, risk_col),
                        height=250, margin=dict(t=40, b=30, l=40, r=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Model agreement analysis
        if all(c in df.columns for c in risk_cols):
            st.markdown("### Model Agreement Analysis")
            df_temp = df.copy()
            df_temp['xgb_flag'] = df_temp['supervised_risk'] > 0.6
            df_temp['iso_flag'] = df_temp['unsupervised_risk'] > 0.6
            df_temp['lstm_flag'] = df_temp['sequence_risk'] > 0.6
            df_temp['agreement_count'] = df_temp[['xgb_flag', 'iso_flag', 'lstm_flag']].sum(axis=1)

            agree_counts = df_temp['agreement_count'].value_counts().sort_index()
            agree_labels = {0: 'All Clear', 1: '1 Model Flags', 2: '2 Models Flag', 3: 'All 3 Flag'}

            fig = px.bar(
                x=[agree_labels.get(int(k), str(k)) for k in agree_counts.index],
                y=agree_counts.values,
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(
                title="How Often Do Models Agree?", height=250,
                xaxis_title="Agreement Level", yaxis_title="Count",
                margin=dict(t=40, b=30, l=40, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collect at least 5 transactions to see model performance analytics.")

# ─────────────────────────────────────────────
# TAB 3: NETWORK GRAPH
# ─────────────────────────────────────────────
with tab3:
    st.subheader("🕸️ Live Transaction Network Graph")
    st.markdown("Visualizes sender → receiver relationships from the live stream. Larger nodes = higher connectivity (potential money laundering hubs). Red = fraud-flagged.")

    if len(st.session_state.transactions) > 2:
        df = pd.DataFrame(st.session_state.transactions)

        if 'nameOrig' in df.columns and 'nameDest' in df.columns:
            G = nx.DiGraph()
            fraud_nodes = set()

            for _, row in df.iterrows():
                sender = row['nameOrig']
                receiver = row['nameDest']
                G.add_edge(sender, receiver)
                if row.get('is_fraud'):
                    fraud_nodes.add(sender)
                    fraud_nodes.add(receiver)

            pos = nx.spring_layout(G, seed=42, k=2.0)

            # Edge traces
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )

            # Node traces
            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                degree = G.degree(node)
                node_text.append(f"{node}<br>Connections: {degree}")
                node_color.append('#ff2b2b' if node in fraud_nodes else '#0068c9')
                node_size.append(max(8, degree * 5))

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers',
                hoverinfo='text', text=node_text,
                marker=dict(size=node_size, color=node_color,
                            line=dict(width=1, color='#fff'))
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                showlegend=False, height=500,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Graph stats
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Unique Nodes", G.number_of_nodes())
            with s2:
                st.metric("Unique Edges", G.number_of_edges())
            with s3:
                st.metric("Fraud-Flagged Nodes", len(fraud_nodes))
    else:
        st.info("Collect at least 3 transactions to build the network graph.")

# ─────────────────────────────────────────────
# TAB 4: MLOPS DRIFT MONITOR
# ─────────────────────────────────────────────
with tab4:
    st.subheader("📊 MLOps: Real-time Data Drift Detection")
    st.markdown("Compares the live streaming transactions against the historical training baseline to detect Concept Drift (e.g., changing consumer behavior, new fraud patterns).")

    if st.button("Generate Drift Report"):
        if len(st.session_state.transactions) > 20:
            with st.spinner("Analyzing statistical distribution shift..."):
                try:
                    ref_path = os.path.abspath(os.path.join(BASE_DIR, "../ml/models_registry/historical_transactions.csv"))
                    ref_df = pd.read_csv(ref_path)
                    curr_df = pd.DataFrame(st.session_state.transactions)

                    result = build_drift_report_html(ref_df, curr_df)
                    st.caption(
                        f"Compared {result.current_rows} live rows against "
                        f"{result.reference_rows} reference rows using: {', '.join(result.columns)}"
                    )
                    components.html(result.html, height=1000, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
        else:
            st.warning("Not enough live transactions gathered yet. Let the stream run a bit longer!")
