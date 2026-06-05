import torch
import torch.nn as nn

class SignLanguageCTCModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_gru_layers=2):
        super(SignLanguageCTCModel, self).__init__()
        
        # 1D CNN without massive MaxPools so we don't aggressively downsample the time dimension.
        # We want sequence length to remain roughly the same, or just half, to keep fine-grained CTC labels.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Bidirectional GRU for CTC
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=hidden_size, 
            num_layers=num_gru_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_gru_layers > 1 else 0.0
        )
        
        # Classifier: num_classes + 1 (for the CTC Blank token)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes + 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        out, _ = self.gru(x)
        logits = self.fc(out)
        
        # CTC loss expects log_softmax
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs
