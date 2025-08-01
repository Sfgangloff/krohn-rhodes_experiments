import unittest
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automaton import Automaton
from generate_traces import generate_traces
from rnn_last_state_predictor import (  # Replace with your actual filename if needed
    AQSequenceDataset,
    collate_batch,
    RNNPredictor,
    predict,
)

states = ["0", "1"]
alphabet =  ["0", "1"]
transitions = {("0,0"): "0",
  ("0,1"): "1",
  ("1,0"): "1",
  ("1,1"): "0"}
initial_state = "0"
accepting_states = ["0"]

automaton = Automaton(states=states,
                      alphabet=alphabet,
                      transitions=transitions,
                      initial_state=initial_state,
                      accepting_states=accepting_states)

traces = generate_traces(10, automaton, num_solutions=100)
data = [("".join(a for (a, _) in trace), trace[-1][1]) for trace in traces]
symbol_to_idx = {a: i + 1 for i, a in enumerate(alphabet)}
idx_to_symbol = {i: a for a, i in symbol_to_idx.items()}
state_to_idx = {q: i for i, q in enumerate(states)}
idx_to_state = {i: q for q, i in state_to_idx.items()}

class TestDFASequenceModel(unittest.TestCase):
    def setUp(self):
        self.dataset = AQSequenceDataset(data, symbol_to_idx, state_to_idx)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
        self.model = RNNPredictor(
            vocab_size=len(symbol_to_idx) + 1,
            num_states=len(state_to_idx),
            embedding_dim=8,
            hidden_dim=16
        )

    def test_data_format(self):
        self.assertTrue(all(isinstance(w, str) and isinstance(q, str) for (w, q) in data))
        self.assertTrue(all(len(q) == 1 for (_, q) in data))

    def test_tokenization_inverse(self):
        for a, idx in symbol_to_idx.items():
            self.assertEqual(a, idx_to_symbol[idx])
        for q, idx in state_to_idx.items():
            self.assertEqual(q, idx_to_state[idx])

    def test_dataset_and_loader(self):
        sample = self.dataset[0]
        self.assertEqual(len(sample), 2)
        self.assertTrue(isinstance(sample[0], torch.Tensor))
        self.assertTrue(isinstance(sample[1], torch.Tensor))
        x, lengths, y = next(iter(self.loader))
        self.assertEqual(x.shape[0], 4)
        self.assertEqual(lengths.shape[0], 4)
        self.assertEqual(y.shape[0], 4)

    def test_model_forward(self):
        x, lengths, y = next(iter(self.loader))
        logits = self.model(x, lengths)
        self.assertEqual(logits.shape, (4, len(state_to_idx)))

    def test_prediction_output(self):
        example = data[0][0]  # string input
        pred = predict(self.model, list(example), symbol_to_idx, idx_to_state)
        self.assertTrue(pred in idx_to_state.values() or pred == "?")

if __name__ == "__main__":
    unittest.main()