from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import groot_bridge_server as bridge  # noqa: E402


class GrootBridgeSeedTest(unittest.TestCase):
    def test_seed_replays_python_numpy_and_torch_rngs(self) -> None:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            with patch.object(torch.cuda, "is_available", return_value=False):
                bridge._seed_inference(43)
                first = (
                    random.random(),
                    float(np.random.random()),
                    torch.rand(3),
                )
                bridge._seed_inference(43)
                second = (
                    random.random(),
                    float(np.random.random()),
                    torch.rand(3),
                )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
