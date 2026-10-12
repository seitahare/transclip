import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clipper


class TestClipper(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"

        self.video_id = "test123"
        self.video_dir = self.input_dir / self.video_id
        self.video_dir.mkdir(parents=True)

        # Dummy video: FFmpeg itself will be mocked.
        (self.video_dir / "source.mp4").touch()

    def save_selection(self, start, end):
        path = self.video_dir / "selection.json"

        path.write_text(
            json.dumps({
                "start": start,
                "end": end
            }),
            encoding="utf-8"
        )

    def test_create_clip(self):
        self.save_selection(1880, 1910)

        with patch.object(clipper, "INPUT_DIR", self.input_dir):
            with patch.object(clipper, "OUTPUT_DIR", self.output_dir):
                with patch("clipper.subprocess.run") as run:
                    result = clipper.create_clip(self.video_id)

        self.assertEqual(result, 0)
        run.assert_called_once()

        command = run.call_args.args[0]

        self.assertEqual(
            command[command.index("-ss") + 1],
            "1880"
        )

        self.assertEqual(
            command[command.index("-t") + 1],
            "30"
        )

    def test_invalid_time_range(self):
        self.save_selection(1910, 1880)

        with patch.object(clipper, "INPUT_DIR", self.input_dir):
            with patch("clipper.subprocess.run") as run:
                result = clipper.create_clip(self.video_id)

        self.assertEqual(result, 1)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
