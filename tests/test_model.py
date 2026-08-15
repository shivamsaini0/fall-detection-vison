"""Tests for the LSTM fall detector model."""

import tempfile
from pathlib import Path

import torch

from models.fall_detector import FallDetector


class TestFallDetector:
    def test_output_shape(self):
        model = FallDetector(input_size=5, hidden_size=64, num_classes=3)
        x = torch.randn(8, 30, 5)  # batch=8, seq_len=30, features=5
        out = model(x)
        assert out.shape == (8, 3)

    def test_single_sample(self):
        model = FallDetector()
        x = torch.randn(1, 30, 5)
        out = model(x)
        assert out.shape == (1, 3)

    def test_predict(self):
        model = FallDetector()
        x = torch.randn(4, 30, 5)
        preds, confs = model.predict(x)
        assert preds.shape == (4,)
        assert confs.shape == (4,)
        assert all(0 <= c <= 1 for c in confs)
        assert all(p in [0, 1, 2] for p in preds)

    def test_gradient_flow(self):
        model = FallDetector()
        x = torch.randn(4, 30, 5)
        y = torch.tensor([0, 1, 2, 0])
        criterion = torch.nn.CrossEntropyLoss()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # Check gradients exist and are not zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_checkpoint_save_load(self):
        model = FallDetector(hidden_size=32)
        x = torch.randn(2, 30, 5)

        # Get prediction before save
        preds_before, _ = model.predict(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pth"
            torch.save({"model_state_dict": model.state_dict()}, path)

            # Load into new model
            model2 = FallDetector(hidden_size=32)
            ckpt = torch.load(path, weights_only=True)
            model2.load_state_dict(ckpt["model_state_dict"])

            preds_after, _ = model2.predict(x)

        assert torch.equal(preds_before, preds_after)

    def test_different_sequence_lengths(self):
        model = FallDetector()
        for seq_len in [10, 30, 60, 100]:
            x = torch.randn(2, seq_len, 5)
            out = model(x)
            assert out.shape == (2, 3)

    def test_parameter_count(self):
        model = FallDetector(input_size=5, hidden_size=128, num_layers=2)
        param_count = sum(p.numel() for p in model.parameters())
        # Should be reasonable for a small LSTM
        assert 100_000 < param_count < 500_000

    def test_class_names(self):
        assert FallDetector.CLASS_NAMES == {0: "normal", 1: "falling", 2: "fallen"}
