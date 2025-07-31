import unittest
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rnn_last_state_predictor import (  # Replace with your actual filename if needed
    DATA,
    symbol_to_idx,
    state_to_idx,
    idx_to_state,
    idx_to_symbol,
    AQSequenceDataset,
    collate_batch,
    RNNPredictor,
    predict,
)

class TestDFASequenceModel(unittest.TestCase):
    def setUp(self):
        self.dataset = AQSequenceDataset(DATA, symbol_to_idx, state_to_idx)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
        self.model = RNNPredictor(
            vocab_size=len(symbol_to_idx) + 1,
            num_states=len(state_to_idx),
            embedding_dim=8,
            hidden_dim=16
        )

    def test_data_format(self):
        self.assertTrue(all(isinstance(w, str) and isinstance(q, str) for (w, q) in DATA))
        self.assertTrue(all(len(q) == 1 for (_, q) in DATA))

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
        example = DATA[0][0]  # string input
        pred = predict(self.model, list(example), symbol_to_idx, idx_to_state)
        self.assertTrue(pred in idx_to_state.values() or pred == "?")

if __name__ == "__main__":
    unittest.main()