import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from automaton import create_automaton_from_yaml
from generate_traces import generate_traces

# ---------------------------------------------------------------------
# Generate (word, final_state) data from DFA traces
# ---------------------------------------------------------------------
AUTOMATON = create_automaton_from_yaml("config.yaml")

TRACES = generate_traces(10, AUTOMATON, num_solutions=100)

# Each trace is a list of (a_i, q_{i-1}), convert to (a1...an, final_state)
DATA = [("".join(a for (a, _) in trace), trace[-1][1]) for trace in TRACES]

# ---------------------------------------------------------------------
# Tokenize symbols and states
# ---------------------------------------------------------------------
alphabet = sorted(set(a for seq, _ in DATA for a in seq))
states = sorted(set(q for _, q in DATA))

symbol_to_idx = {a: i + 1 for i, a in enumerate(alphabet)}  # +1 to reserve 0 for padding
idx_to_symbol = {i: a for a, i in symbol_to_idx.items()}
state_to_idx = {q: i for i, q in enumerate(states)}
idx_to_state = {i: q for q, i in state_to_idx.items()}

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class AQSequenceDataset(Dataset):
    def __init__(self, data, symbol_to_idx, state_to_idx):
        self.encoded_data = [
            (torch.tensor([symbol_to_idx[a] for a in prefix], dtype=torch.long),
             torch.tensor(state_to_idx[target], dtype=torch.long))
            for prefix, target in data
        ]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

def collate_batch(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded, lengths, targets

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class RNNPredictor(nn.Module):
    def __init__(self, vocab_size, num_states, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_states)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.rnn(packed)
        out = self.fc(hn[-1])
        return out

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_model(model, dataloader, num_epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for x, lengths, y in dataloader:
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
def predict(model, prefix, symbol_to_idx, idx_to_state):
    model.eval()
    with torch.no_grad():
        if not prefix:
            x = torch.tensor([[0]], dtype=torch.long)
            lengths = torch.tensor([0])
        else:
            indices = [symbol_to_idx[a] for a in prefix]
            x = torch.tensor([indices], dtype=torch.long)
            lengths = torch.tensor([len(indices)])
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        return idx_to_state.get(pred_idx, "?")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    dataset = AQSequenceDataset(DATA, symbol_to_idx, state_to_idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

    model = RNNPredictor(vocab_size=len(symbol_to_idx)+1, num_states=len(state_to_idx))
    train_model(model, loader, num_epochs=30)

    # Test predictions
    test_prefixes = [
        list("010"),
        list("111"),
        list("0"),
        list("1010")
    ]

    print("\nPredictions:")
    for prefix in test_prefixes:
        pred = predict(model, prefix, symbol_to_idx, idx_to_state)
        print(f"{prefix} → {pred}")