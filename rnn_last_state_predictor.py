"""
rnn_last_state_predictor.py

Train an RNN to simulate the behavior of a deterministic finite automaton (DFA) by predicting its final state
given a sequence of input symbols.

Pipeline Overview:
------------------
1. Generate valid execution traces from a DFA using a SAT solver.
2. Convert each trace [(a₁, q₀), ..., (aₙ, qₙ₋₁)] into a training example:
       ("a₁a₂...aₙ", qₙ)
3. Tokenize the input alphabet and state set.
4. Train an LSTM-based sequence model to predict the final state.
5. Evaluate the model's prediction on new input sequences.

Dependencies:
-------------
- PyTorch (model, training)
- automaton.py (defines DFA structure)
- generate_traces.py (uses PySAT to generate DFA-consistent traces)

Output:
-------
The script prints the training loss and predictions on sample inputs after training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from automaton import create_automaton_from_yaml
from generate_traces import generate_traces, chain_transition
from automaton import load_automata_and_output_maps, cascade_multiple_automata, flatten_dfa_states

# TODO: The original article extracts the automaton from hidden states. When we do this extraction for a cascade automaton, will the extraction give 
# a decomposition of the automaton? 
# TODO: Is performance/length generalization improved when using the cascade decomposition ? 

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class AQSequenceDataset(Dataset):
    """
    PyTorch Dataset that encodes DFA sequences and their final states.

    Each input is a list of input symbols (as indices).
    Each target is the index of the final state reached by the DFA.
    """
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
    """
    Collate function for padding variable-length input sequences.

    Args:
        batch: List of (sequence_tensor, target_tensor) pairs.

    Returns:
        padded (Tensor): batch_size × max_len tensor with padded sequences.
        lengths (Tensor): Original lengths of each sequence.
        targets (Tensor): Batch of final state indices.
    """
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded, lengths, targets

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class RNNPredictor(nn.Module):
    """
    Recurrent model (LSTM) that embeds input symbols, processes them, and predicts the final DFA state.

    Args:
        vocab_size (int): Number of distinct input symbols (+1 for padding).
        num_states (int): Number of DFA states (classification classes).
        embedding_dim (int): Size of the symbol embedding vectors.
        hidden_dim (int): Size of the LSTM hidden state.
    """
    def __init__(self, vocab_size, num_states, embedding_dim=32, hidden_dim=64):
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
    """
    Train the model using cross-entropy loss.

    Args:
        model (nn.Module): The RNN classifier.
        dataloader (DataLoader): Batches of (sequence, length, target) triples.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
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
    """
    Predict the final DFA state given an input sequence of symbols.

    Args:
        model (nn.Module): Trained model.
        prefix (List[str]): Sequence of input symbols (e.g., ['0', '1']).
        symbol_to_idx (dict): Maps symbols to indices.
        idx_to_state (dict): Maps predicted index to DFA state string.

    Returns:
        str: Predicted final state.
    """
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

    # ---------------------------------------------------------------------
    # Generate (word, final_state) data from DFA traces
    # ---------------------------------------------------------------------
    import yaml

    MULTIPLE_AUTOMATA = True

    if not MULTIPLE_AUTOMATA: 

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        STATES = set(config["states"])
        ALPHABET = set(config["alphabet"])
        INITIAL_STATE = config["initial_state"]
        ACCEPTING_STATES = set(config["accepting_states"])

        AUTOMATON = create_automaton_from_yaml("config.yaml")
    else: 

        automata, output_maps = load_automata_and_output_maps("multiple_config.yaml")

        cascade = cascade_multiple_automata(automata, output_maps)
        AUTOMATON = flatten_dfa_states(cascade)
        ALPHABET = AUTOMATON.alphabet
        STATES = AUTOMATON.states


    TRACES = generate_traces(10, AUTOMATON, num_solutions=1024)

    # Convert full trace to (input string, final state)
    DATA = [("".join(a for (a, _) in trace), trace[-1][1]) for trace in TRACES]
    # ---------------------------------------------------------------------
    # Tokenize symbols and states
    # ---------------------------------------------------------------------

    SYMBOL_TO_IDX = {a: i + 1 for i, a in enumerate(ALPHABET)}  # index 0 is reserved for padding
    IDX_TO_SYMBOL = {i: a for a, i in SYMBOL_TO_IDX.items()}
    STATE_TO_IDX = {q: i for i, q in enumerate(STATES)}
    IDX_TO_STATE = {i: q for q, i in STATE_TO_IDX.items()}


    dataset = AQSequenceDataset(DATA, SYMBOL_TO_IDX, STATE_TO_IDX)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

    model = RNNPredictor(vocab_size=len(SYMBOL_TO_IDX)+1, num_states=len(STATE_TO_IDX))
    train_model(model, loader, num_epochs=30)

    torch.save(model.state_dict(), "models/rnn_dfa_model.pt")

    # Test predictions
    test_prefixes = [
        list("abaabbbaab"),
        list("bbabbaaaaa"),
        list("aabbaaaaab"),
        list("baaaababaa")
    ]
    ground_truth = []
    for test in test_prefixes: 
        ground_truth.append(chain_transition(inputs=test,automaton=AUTOMATON))

    print("\nPredictions:")
    for prefix in test_prefixes:
        pred = predict(model, prefix, SYMBOL_TO_IDX, IDX_TO_STATE)
        print(f"{prefix} → {pred}")
    print(ground_truth)