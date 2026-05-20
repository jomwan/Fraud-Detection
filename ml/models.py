import torch
import torch.nn as nn

class FraudLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # Pass the output of the last sequence step to the Linear layer
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)
