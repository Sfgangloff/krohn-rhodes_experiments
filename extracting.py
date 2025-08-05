import torch
from typing import List
from rnn_last_state_predictor import RNNPredictor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import yaml 
from automaton import create_automaton_from_yaml
from automaton import load_automata_and_output_maps, cascade_multiple_automata, flatten_dfa_states

def get_hidden_state(model: RNNPredictor, 
                     prefix: List[str], 
                     symbol_to_idx: dict) -> torch.Tensor:
    """
    Return the hidden state of the RNN after processing a given input sequence.

    Args:
        model (RNNPredictor): Trained RNN model.
        prefix (List[str]): Sequence of input symbols (e.g., ['a', 'b']).
        symbol_to_idx (dict): Mapping from input symbols to indices.

    Returns:
        Tensor: Final hidden state vector (1D tensor).
    """
    model.eval()
    with torch.no_grad():
        if not prefix:
            x = torch.tensor([[0]], dtype=torch.long)  # padded input
            lengths = torch.tensor([0])
        else:
            indices = [symbol_to_idx[a] for a in prefix]
            x = torch.tensor([indices], dtype=torch.long)
            lengths = torch.tensor([len(indices)])

        emb = model.embedding(x)  # shape: (1, seq_len, emb_dim)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = model.rnn(packed)  # hn: (num_layers, batch=1, hidden_dim)
        return hn[-1, 0]  # shape: (hidden_dim,)
    
if __name__ == "__main__":
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

    SYMBOL_TO_IDX = {a: i + 1 for i, a in enumerate(ALPHABET)}  # index 0 is reserved for padding
    STATE_TO_IDX = {q: i for i, q in enumerate(STATES)}
    model = RNNPredictor(vocab_size=len(SYMBOL_TO_IDX)+1, num_states=len(STATE_TO_IDX))
    model.load_state_dict(torch.load("models/rnn_dfa_model.pt"))
    hidden = get_hidden_state(model, list("abba"), SYMBOL_TO_IDX)
    print(hidden.shape)