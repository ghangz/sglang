import unittest
from unittest.mock import patch

from sglang.check_env import GPUEnv, HIPEnv


class TestCheckEnv(unittest.TestCase):
    @patch("sglang.check_env.subprocess.check_output", side_effect=FileNotFoundError)
    def test_nvcc_missing_binary_returns_not_available(self, _):
        self.assertEqual(GPUEnv()._get_nvcc_info(), {"NVCC": "Not Available"})

    @patch("sglang.check_env.subprocess.check_output", side_effect=FileNotFoundError)
    def test_hipcc_missing_binary_returns_not_available(self, _):
        self.assertEqual(HIPEnv()._get_hipcc_info(), {"HIPCC": "Not Available"})

    @patch("torch.utils.cpp_extension.ROCM_HOME", None)
    def test_hipcc_missing_rocm_home_returns_not_available(self):
        self.assertEqual(HIPEnv()._get_hipcc_info(), {"HIPCC": "Not Available"})


if __name__ == "__main__":
    unittest.main()
