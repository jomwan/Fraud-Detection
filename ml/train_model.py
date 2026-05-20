import logging
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import networkx as nx
import joblib
import os
import sys

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.evaluate_model import generate_evaluation_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_registry')

def build_graph_features(df):
    """Computes network intelligence features using Graph Analytics."""
    df = df.copy()
    logger.info("Building Transaction Graph...")
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
        logger.error("Historical data not found at %s. Run data_generator.py first.", historical_path)
        return

    logger.info("Loading historical data from %s...", historical_path)
    df = pd.read_csv(historical_path)

    df, G, centrality_map = build_graph_features(df)

    logger.info("Preprocessing data...")
    X, y, le_type = preprocess_data(df)
    stratify_target = y if y.nunique() > 1 else None
    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )

    logger.info("Training Supervised XGBoost Model...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    logger.info("Training Unsupervised Isolation Forest Model...")
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X_train)

    logger.info("Generating holdout model evaluation report...")
    generate_evaluation_report(
        xgb_model=xgb_model,
        iso_model=iso_model,
        X_eval=X_eval,
        y_eval=y_eval,
        output_path=os.path.join(MODELS_DIR, 'evaluation_report.json'),
        dataset_name="holdout_20_percent"
    )

    logger.info("Saving models and graph metadata to %s...", MODELS_DIR)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    joblib.dump(iso_model, os.path.join(MODELS_DIR, 'iso_model.pkl'))
    joblib.dump(le_type, os.path.join(MODELS_DIR, 'le_type.pkl'))
    joblib.dump(centrality_map, os.path.join(MODELS_DIR, 'graph_centrality.pkl'))

    logger.info("Model Training Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
