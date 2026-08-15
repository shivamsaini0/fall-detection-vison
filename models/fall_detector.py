"""LSTM-based fall detection classifier."""

import torch
import torch.nn as nn


class FallDetector(nn.Module):
    """LSTM classifier for fall detection from skeleton feature sequences.

    Input: (batch, seq_len, input_size) — temporal sequences of skeleton features.
    Output: (batch, num_classes) — class logits for normal/falling/fallen.
    """

    CLASS_NAMES = {0: "normal", 1: "falling", 2: "fallen"}

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)
        return self.fc(out)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict class and confidence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Tuple of (predicted_classes, confidences).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        return predictions, confidences
