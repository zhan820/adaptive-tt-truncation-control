"""Behavioral checks for variable-size and joint-action controller inputs."""

import tempfile
import unittest
from pathlib import Path

import torch

from adaptive_tt_dgmres.model import (
    JointControllerNet,
    load_controller_checkpoint,
    save_controller_checkpoint,
    tensorize_histories,
)


def state(interfaces, offset=0.0):
    return {
        "tokens": (torch.arange(interfaces * 10).reshape(interfaces, 10) / 100 + offset).tolist(),
        "global": (torch.arange(18) / 20 + offset).tolist(),
    }


class JointControllerModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(417)
        self.model = JointControllerNet().eval()
        self.actions = [[0.25, -2.0], [0.50, -1.0], [1.0, 0.0]]
        self.histories = [[state(2), state(3, 0.1)], [state(1, 0.2)]]

    def test_padding_does_not_change_predictions(self):
        inputs = tensorize_histories(self.histories, self.actions)
        expected = self.model(**inputs)
        # Corrupt both interface padding and the second example's time padding.
        inputs["tokens"][~inputs["token_mask"]] = float("nan")
        inputs["globals"][1, 1] = float("nan")
        actual = self.model(**inputs)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        for index, history in enumerate(self.histories):
            alone = self.model(**tensorize_histories([history], self.actions))
            torch.testing.assert_close(actual["log_cost"][index], alone["log_cost"][0])

    def test_history_limit_and_per_example_actions(self):
        history = [state(2, float(i)) for i in range(6)]
        limited = tensorize_histories([history], self.actions, history_limit=3)
        explicit = tensorize_histories([history[-3:]], [self.actions])
        self.assertEqual(limited["lengths"].tolist(), [3])
        torch.testing.assert_close(self.model(**limited)["q"], self.model(**explicit)["q"])

    def test_actions_affect_predictions_and_training_gradients_are_finite(self):
        inputs = tensorize_histories(self.histories, self.actions, dtype=torch.float64)
        result = self.model(**inputs)
        self.assertEqual(result["log_cost"].shape, (2, 3))
        self.assertIs(result["q"], result["log_cost"])
        self.assertFalse(torch.allclose(result["q"][:, 0], result["q"][:, 2]))
        loss = sum(result[key].square().mean() for key in (
            "log_cost", "cycle_log_seconds", "log_reduction", "log_rank"))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_preserves_predictions_dtype_and_metadata(self):
        self.model.double()
        inputs = tensorize_histories(self.histories, self.actions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.pt"
            save_controller_checkpoint(path, self.model, {"history_limit": 5, "actions": self.actions})
            restored, metadata = load_controller_checkpoint(path)
        self.assertEqual(metadata["actions"], self.actions)
        self.assertEqual(next(restored.parameters()).dtype, torch.float64)
        self.assertFalse(restored.training)
        for key, expected in self.model(**inputs).items():
            torch.testing.assert_close(restored(**inputs)[key], expected)

    def test_invalid_actual_steps_are_rejected(self):
        inputs = tensorize_histories(self.histories, self.actions)
        inputs["token_mask"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "actual time step"):
            self.model(**inputs)
        inputs = tensorize_histories(self.histories, self.actions)
        inputs["lengths"][0] = 0
        with self.assertRaisesRegex(ValueError, "history length"):
            self.model(**inputs)
        with self.assertRaisesRegex(ValueError, "empty histories"):
            tensorize_histories([[]], self.actions)


if __name__ == "__main__":
    unittest.main()
