import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_registry')

def build_graph_features(df):
    """Computes network intelligence features using Graph Analytics."""
    print("Building Transaction Graph...")
    G = nx.from_pandas_edgelist(df, 'nameOrig', 'nameDest', create_using=nx.Graph())
    centrality = nx.degree_centrality(G)
    df['sender_degree'] = df['nameOrig'].map(centrality).fillna(0)
    df['receiver_degree'] = df['nameDest'].map(centrality).fillna(0)
    return df, G, centrality

def preprocess_data(df):
    """Preprocesses using native PaySim columns including balance features."""
    df = df.copy()
    le_type = LabelEncoder()
    df['type_encoded'] = le_type.fit_transform(df['type'])

    features = ['amount', 'type_encoded', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'sender_degree', 'receiver_degree']
    X = df[features]
    y = df['isFraud']
    return X, y, le_type

def train():
    historical_path = os.path.join(MODELS_DIR, "historical_transactions.csv")
    if not os.path.exists(historical_path):
        print(f"Historical data not found at {historical_path}. Run data_generator.py first.")
        return

    print(f"Loading historical data from {historical_path}...")
    df = pd.read_csv(historical_path)

    df, G, centrality_map = build_graph_features(df)

    print("Preprocessing data...")
    X, y, le_type = preprocess_data(df)

    print("Training Supervised XGBoost Model...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)

    print("Training Unsupervised Isolation Forest Model...")
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X)

    print(f"Saving models and graph metadata to {MODELS_DIR}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    joblib.dump(iso_model, os.path.join(MODELS_DIR, 'iso_model.pkl'))
    joblib.dump(le_type, os.path.join(MODELS_DIR, 'le_type.pkl'))
    joblib.dump(centrality_map, os.path.join(MODELS_DIR, 'graph_centrality.pkl'))

    print("\nModel Training Complete.")

if __name__ == "__main__":
    train()
