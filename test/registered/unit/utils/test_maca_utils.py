import importlib.util
import types
import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[4]
MACA_UTILS_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "utils" / "maca.py"

spec = importlib.util.spec_from_file_location("sglang_srt_utils_maca", MACA_UTILS_PATH)
maca_utils = importlib.util.module_from_spec(spec)
sys.modules["sglang_srt_utils_maca"] = maca_utils
assert spec.loader is not None
spec.loader.exec_module(maca_utils)


class MacaUtilsTest(unittest.TestCase):
    def test_get_torch_maca_version(self):
        torch_module = types.SimpleNamespace(version=types.SimpleNamespace(maca="2.27"))

        self.assertEqual(maca_utils.get_torch_maca_version(torch_module), "2.27")

    def test_has_maca_toolkit_env_detects_cu_bridge(self):
        with TemporaryDirectory() as tmp_dir:
            maca_path = Path(tmp_dir) / "maca"
            (maca_path / "tools" / "cu-bridge").mkdir(parents=True)

            self.assertTrue(maca_utils.has_maca_toolkit_env({"MACA_PATH": str(maca_path)}))

    def test_is_maca_available_requires_device_when_using_env_fallback(self):
        with TemporaryDirectory() as tmp_dir:
            maca_path = Path(tmp_dir) / "maca"
            (maca_path / "lib").mkdir(parents=True)
            cuda = types.SimpleNamespace(is_available=lambda: True)
            torch_module = types.SimpleNamespace(
                version=types.SimpleNamespace(),
                cuda=cuda,
            )

            self.assertTrue(maca_utils.is_maca_available(torch_module, {"MACA_PATH": str(maca_path)}))


if __name__ == "__main__":
    unittest.main()
