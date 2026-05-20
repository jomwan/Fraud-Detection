import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from confluent_kafka import Consumer
from evidently import Report
from evidently.presets import DataDriftPreset
import os
import sys

# Adjust Python Path to resolve local imports cleanly from workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        st.subheader("Live Transaction Ledger & Alerting Engine")

        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)

            def highlight_fraud(row):
                if row.get('is_fraud'):
                    return ['background-color: rgba(255, 50, 50, 0.3)'] * len(row)
                return [''] * len(row)

            display_cols = [
                'nameOrig', 'nameDest', 'type', 'amount', 'velocity_5m',
                'supervised_risk', 'unsupervised_risk', 'sequence_risk',
                'combined_risk', 'reason', 'action'
            ]
            existing_cols = [c for c in display_cols if c in df.columns]

            fmt = {"amount": "${:.2f}"}
            for rc in ['supervised_risk', 'unsupervised_risk', 'sequence_risk', 'combined_risk']:
                if rc in existing_cols:
                    fmt[rc] = "{:.3f}"

            st.dataframe(
                df[existing_cols].head(15).style.apply(highlight_fraud, axis=1).format(fmt),
                width='stretch'
            )

            # Risk timeline scatter
            if len(df) > 1 and 'action' in df.columns:
                df['color'] = df['action'].map({'BLOCK': '#ff2b2b', 'ALLOW': '#0068c9'}).fillna('#0068c9')
                df['tx_index'] = range(len(df))
                st.scatter_chart(df, x='tx_index', y='combined_risk', color='color')

            # Amount distribution histogram
            if len(df) > 3 and 'amount' in df.columns:
                st.markdown("**💰 Transaction Amount Distribution** _(look for clusters near $10K = smurfing)_")
                fig = px.histogram(df, x='amount', nbins=30, color_discrete_sequence=['#0068c9'])
                fig.update_layout(height=250, margin=dict(t=10, b=30, l=40, r=10))
                st.plotly_chart(fig, width='stretch')

with tab1:
    live_dashboard()

# ─────────────────────────────────────────────
# TAB 2: MODEL PERFORMANCE
# ─────────────────────────────────────────────
with tab2:
    st.subheader("📈 Model Performance & Contribution Analysis")

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
                st.plotly_chart(fig, width='stretch')
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
                    st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')
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
            st.plotly_chart(fig, width='stretch')

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
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                    ref_path = os.path.abspath(os.path.join(BASE_DIR, "../ml/models_registry/historical_transactions.csv"))
                    ref_df = pd.read_csv(ref_path)
                    curr_df = pd.DataFrame(st.session_state.transactions)

                    # Compare all columns shared between reference and live data
                    candidate_cols = ['amount', 'velocity_5m', 'supervised_risk', 'unsupervised_risk', 'sequence_risk']
                    shared_cols = [c for c in candidate_cols if c in ref_df.columns and c in curr_df.columns]
                    if not shared_cols:
                        shared_cols = ['amount']

                    report = Report(metrics=[DataDriftPreset()])
                    snapshot = report.run(reference_data=ref_df[shared_cols], current_data=curr_df[shared_cols])

                    report_html = snapshot.get_html_str(as_iframe=False)
                    st.html(f'<div style="height:1000px;overflow-y:auto">{report_html}</div>')
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
        else:
            st.warning("Not enough live transactions gathered yet. Let the stream run a bit longer!")
