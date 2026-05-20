import unittest

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.test_utils import CustomTestCase


class _DummyLayer(torch.nn.Module):
    params_dtype = torch.bfloat16


class TestHummingMoeRunner(CustomTestCase):
    def test_none_a2a_reuses_runner_core(self):
        old_backend = moe_utils.MOE_A2A_BACKEND
        try:
            moe_utils.MOE_A2A_BACKEND = MoeA2ABackend.NONE
            runner = MoeRunner(
                MoeRunnerBackend.HUMMING,
                MoeRunnerConfig(
                    num_experts=1,
                    num_local_experts=1,
                    layer=_DummyLayer(),
                ),
            )
            self.assertIsNone(runner.fused_func)
            self.assertIsNotNone(runner.runner_core)
        finally:
            moe_utils.MOE_A2A_BACKEND = old_backend


if __name__ == "__main__":
    unittest.main()
