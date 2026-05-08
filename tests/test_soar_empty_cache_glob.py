import unittest
import types
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("av", types.ModuleType("av"))

from musubi_tuner.dataset.image_video_dataset import _glob_latent_cache_files  # noqa: E402


class TestSoarEmptyCacheGlob(unittest.TestCase):
    def test_legacy_empty_prompt_cache_is_not_treated_as_latent_cache(self):
        with TemporaryDirectory() as temp_dir:
            valid_latent = Path(temp_dir) / "image_1024x1024_zi.safetensors"
            legacy_empty = Path(temp_dir) / "__soar_empty_prompt_zi.safetensors"
            new_empty_dir = Path(temp_dir) / "__soar_empty_prompt"
            new_empty = new_empty_dir / "zi.safetensors"
            new_empty_dir.mkdir()

            valid_latent.write_bytes(b"")
            legacy_empty.write_bytes(b"")
            new_empty.write_bytes(b"")

            cache_files = {Path(path).name for path in _glob_latent_cache_files(temp_dir, "zi")}

        self.assertEqual(cache_files, {valid_latent.name})


if __name__ == "__main__":
    unittest.main()
