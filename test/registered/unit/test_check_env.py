import unittest
from unittest.mock import patch

from sglang.check_env import GPUEnv, HIPEnv, MUSAEnv, NPUEnv


class TestCheckEnv(unittest.TestCase):
    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_gpu_topology_missing_command_returns_empty_info(self, _):
        self.assertEqual(GPUEnv().get_topology(), {})

    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_hip_topology_missing_command_returns_empty_info(self, _):
        self.assertEqual(HIPEnv().get_topology(), {})

    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_npu_topology_missing_command_returns_empty_info(self, _):
        self.assertEqual(NPUEnv().get_topology(), {})

    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_musa_topology_missing_command_returns_empty_info(self, _):
        self.assertEqual(MUSAEnv().get_topology(), {})


if __name__ == "__main__":
    unittest.main()
