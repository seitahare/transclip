import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitles import ensure_transcript, subtitle_candidates


class TestSubtitles(unittest.TestCase):

    def test_manual_subtitles_have_priority(self):
        metadata = {
            "subtitles": {"en": []},
            "automatic_captions": {"en-orig": []},
        }

        candidates = subtitle_candidates(metadata)

        self.assertEqual(
            candidates,
            [
                ("youtube_manual", "en"),
                ("youtube_auto", "en-orig"),
            ],
        )

    def test_download_failure_falls_back_to_whisper(self):
        with tempfile.TemporaryDirectory() as temp:
            video_dir = Path(temp)

            metadata = {
                "subtitles": {"en": []},
                "automatic_captions": {"en-orig": []},
            }

            (video_dir / "source.info.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            transcript_path = video_dir / "transcript.json"

            # Simulate failed subtitle downloads.
            failed_download = subprocess.CompletedProcess(
                args=[], returncode=1
            )

            with patch(
                "subtitles.subprocess.run",
                return_value=failed_download,
            ) as download:
                with patch(
                    "subtitles.transcribe_with_whisper",
                    return_value=transcript_path,
                ) as whisper:

                    result = ensure_transcript(
                        video_dir,
                        url="https://www.youtube.com/watch?v=XuoqKYxDHVc",
                    )

            self.assertEqual(result, transcript_path)
            self.assertEqual(download.call_count, 2)
            whisper.assert_called_once_with(video_dir)


if __name__ == "__main__":
    unittest.main()
