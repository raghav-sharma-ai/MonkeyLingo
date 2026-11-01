import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class AudioIngest:
    """
    Handles audio extraction and splitting from video using ffmpeg and Whisper.
    """
    def __init__(self, video_path: str, output_dir: str = "data/scene"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.master_wav = self.output_dir / "master_audio_48k.wav"

    def extract_audio(self) -> Path:
        """
        Extracts 48kHz mono 16-bit PCM WAV audio from source video.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            str(self.master_wav)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {res.stderr.decode('utf-8')}")
        return self.master_wav

    def segment_with_whisper(self, model_size: str = "base") -> List[Dict[str, Any]]:
        """
        Transcribes dialogue and generates timestamped segment intervals.
        """
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(str(self.master_wav), word_timestamps=True)
        
        segments = []
        slices_dir = self.output_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, seg in enumerate(result.get("segments", [])):
            start = round(seg["start"], 3)
            end = round(seg["end"], 3)
            duration = round(end - start, 3)
            text = seg["text"].strip()
            
            if duration < 0.3 or not text:
                continue

            slice_path = slices_dir / f"line_{idx:03d}.wav"
            # Extract line audio with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(self.master_wav),
                "-c", "copy",
                str(slice_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            segments.append({
                "id": f"line_{idx:03d}",
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "slice_path": str(slice_path)
            })
            
        return segments
