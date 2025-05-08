import torch.nn as nn


# -------- LSTM with Residuals (something like ResNet) --------
class LSTMResidualModel(nn.Module):
    def __init__(
        self, input_size=1, hidden_size=64, num_layers=2, 
        dropout=0.3, target_len=96):
        
        super(LSTMResidualModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, target_len)
        self.residual = nn.Linear(input_size, target_len)  # 1: MCP input

    def forward(self, x):
        out, _ = self.lstm(x)  # lstm outputs: output y, (hidden state, cell state)
        last_hidden = out[:, -1, :]  # to get the last hidden state
        prediction = self.linear(last_hidden)

        # Residual connection from last input time step
        residual = self.residual(x[:, -1, :])
        return prediction + residual
