import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from ml.models import FraudLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_registry')

def build_sequences_from_csv(csv_path, seq_len=5):
    """Build real transaction sequences from historical data, grouped by sender."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(['nameOrig', 'step'])

    sequences = []
    labels = []

    for sender_id, group in df.groupby('nameOrig'):
        amounts = group['amount'].values

        if len(amounts) < seq_len:
            continue

        # Create sliding windows of seq_len
        for i in range(len(amounts) - seq_len + 1):
            seq = amounts[i:i + seq_len]

            # Label as smurfing if 3+ amounts are in the $9,000-$10,000 range
            # (structuring to avoid $10k reporting threshold)
            smurf_count = sum(1 for a in seq if 9000 <= a <= 10000)
            is_smurf = 1.0 if smurf_count >= 3 else 0.0

            sequences.append(seq)
            labels.append(is_smurf)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_lstm():
    csv_path = os.path.join(MODELS_DIR, "historical_transactions.csv")

    if os.path.exists(csv_path):
        print(f"Building sequences from historical transaction data at {csv_path}...")
        sequences, labels = build_sequences_from_csv(csv_path, seq_len=5)
        print(f"  Built {len(sequences)} sequences ({int(sum(labels))} positive, {int(len(labels) - sum(labels))} negative)")
    else:
        print(f"WARNING: {csv_path} not found. Using synthetic fallback.")
        sequences = np.random.uniform(10, 500, (1000, 5)).astype(np.float32)
        labels = np.zeros(1000, dtype=np.float32)

    # Augment with explicit smurfing patterns to ensure the model learns them
    print("Augmenting with synthetic smurfing patterns...")
    smurf_seqs = np.random.uniform(9000, 9999, (500, 5)).astype(np.float32)
    smurf_labels = np.ones(500, dtype=np.float32)

    # Also add clean normal sequences to balance
    normal_seqs = np.random.uniform(10, 500, (500, 5)).astype(np.float32)
    normal_labels = np.zeros(500, dtype=np.float32)

    # Ensure sequences is 2D even if empty (PaySim has high sender cardinality)
    if len(sequences) == 0 or sequences.ndim == 1:
        sequences = np.empty((0, 5), dtype=np.float32)
        labels = np.empty(0, dtype=np.float32)

    sequences = np.concatenate([sequences, smurf_seqs, normal_seqs])
    labels = np.concatenate([labels, smurf_labels, normal_labels])

    # Clip at $10K and normalize for LSTM input (batch, seq_len, features)
    sequences = np.clip(sequences, 0, 10000)
    X = (sequences / 10000.0).reshape(-1, 5, 1)
    y = labels.reshape(-1, 1)

    # Reproducible shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Train/validation split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)} samples | Validation: {len(X_val)} samples")

    # Create DataLoaders for mini-batch training
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = FraudLSTM(input_size=1, hidden_size=16)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Training PyTorch LSTM on {len(X_train)} sequences for 30 epochs (batch_size=64)...")
    for epoch in range(30):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = train_loss / max(train_batches, 1)

        # --- Validation ---
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"  Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")

    lstm_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
    print(f"Saving LSTM Model to '{lstm_path}'...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), lstm_path)
    print("LSTM Training Complete!")

if __name__ == '__main__':
    train_lstm()
