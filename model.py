import torch
import torch.nn as nn
import torch.nn.functional as F


# Apply dropout to the sequence position, not a neuron/feature ==> Ignore certain amino acids.
class DropoutSeqPos(nn.Module):
    def __init__(self, p):
        super(DropoutSeqPos, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        mask = (torch.rand(x.size(0), x.size(1), device=x.device) > self.p).float() #generates a random mask for sequence if training
        return x * mask.unsqueeze(-1)

### Attention mechanism: Focus on the most relevant parts of the sequence
class DeepLocAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, align_dim, decode_steps):
        super(DeepLocAttention, self).__init__()
        self.decode_steps = decode_steps
        self.hidden_dim = hidden_dim

        self.W_align = nn.Linear(hidden_dim, align_dim, bias=False)
        self.U_align = nn.Linear(input_dim, align_dim, bias=False)
        self.v_align = nn.Linear(align_dim, 1, bias=False)

        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, encoded, mask):
        batch_size, seq_len, feat_dim = encoded.size()
        device = encoded.device

        h_t = torch.zeros(batch_size, self.hidden_dim).to(device) # Attention weights
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

        alpha_list = []
        context_list = []

        U_align_out = self.U_align(encoded)

        for _ in range(self.decode_steps):
            W_align_out = self.W_align(h_t).unsqueeze(1)
            scores = self.v_align(torch.tanh(W_align_out + U_align_out)).squeeze(-1) # Scores for each position

            scores = scores.masked_fill(mask == 0, float('-inf'))
            alpha = torch.softmax(scores, dim=1) # how much attention should be given to each position
            context = torch.sum(encoded * alpha.unsqueeze(-1), dim=1) # Use attention weights to get weighted context vector 

            h_t, c_t = self.lstm_cell(context, (h_t, c_t)) # Pass the context vector to the LSTM cell
            alpha_list.append(alpha)
            context_list.append(context)

        alphas = torch.stack(alpha_list, dim=1) # Attention weights
        contexts = torch.stack(context_list, dim=1)

        return alphas, contexts

class DeepLocModel(nn.Module):
    def __init__(self, n_feat, n_class, n_hid=256, n_filt=10, drop_per=0.2, drop_hid=0.5):
        super(DeepLocModel, self).__init__()
        self.drop_seq = DropoutSeqPos(drop_per)

        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(n_feat, n_filt, kernel_size=k, padding=k // 2)
            for k in [1, 3, 5, 9, 15, 21]
        ]) # Different filters as stated in paper
        self.cnn_out = nn.Conv1d(n_filt * 6, 64, kernel_size=3, padding=1) # Reduce dimensionality

        self.blstm = nn.LSTM(64, n_hid, bidirectional=True, batch_first=True)
        self.attention = DeepLocAttention(n_hid * 2, n_hid * 2, n_hid, decode_steps=10)
        # Correction: DeepLoc 1.0 uses the tree only to explain predictions, not to make them.
        self.output_layer = nn.Linear(n_hid * 2, 10)
        self.fc_membrane = nn.Linear(n_hid * 2, 1)  # Membrane-bound classifier (binary)
        self.drop_hid = nn.Dropout(drop_hid)

    def forward(self, x, mask):
        x = self.drop_seq(x)
        x = x.transpose(1, 2)
        x = torch.cat([F.relu(conv(x)) for conv in self.cnn_layers], dim=1)
        x = F.relu(self.cnn_out(x))
        x = x.transpose(1, 2)
        x = self.drop_hid(x)

        packed_out, _ = self.blstm(x)

        alphas, contexts = self.attention(packed_out, mask)
        last_context = contexts[:, -1, :]

        # Membrane-bound vs soluble prediction (binary classifier)
        membrane_out = torch.sigmoid(self.fc_membrane(self.drop_hid(last_context))) 

        out = self.output_layer(last_context)  # Predict localization
        return out, alphas, last_context, membrane_out
