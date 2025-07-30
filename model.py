import torch
import torch.nn as nn
from transformers import AutoModel

class DeBERTaLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_labels=2):
        super().__init__()

        self.deberta = AutoModel.from_pretrained("microsoft/deberta-base")
        for param in self.deberta.parameters():
            param.requires_grad = False  # freeze DeBERTa (as we don't have enough resources, we will not train DeBERTa in this model)

        self.lstm = nn.LSTM(
            input_size=self.deberta.config.hidden_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)

        lstm_out, _ = self.lstm(outputs.last_hidden_state)  # shape: [batch, seq_len, hidden*2]
        final_hidden = lstm_out[:, -1, :]  # last token output
        logits = self.fc(final_hidden)
        return logits