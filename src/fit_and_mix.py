import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import soundfile as sf
import numpy as np

class AudioFitter:
    """
    Time-fitting, pitch modulation, and finishing layer:
    - Pitch nudges via rubberband to avoid flat/monotonic TTS.
    - Time-fitting via rubberband within safe boundaries.
    - Micro-crossfades to eliminate clicks and robotic start/stops.
    - Intelligent audio mixdown preserving background room tone.
    """
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def get_duration(self, audio_path: str) -> float:
        info = sf.info(audio_path)
        return info.duration

    def process_take(
        self,
        input_wav: str,
        target_duration: float,
        pitch_semitones: float,
        output_wav: str,
        max_stretch_ratio: float = 0.15
    ) -> float:
        """
        Applies pitch modulation + time fitting using rubberband.
        """
        curr_dur = self.get_duration(input_wav)
        ratio = curr_dur / target_duration
        clamped_ratio = max(1.0 - max_stretch_ratio, min(1.0 + max_stretch_ratio, ratio))
        time_factor = 1.0 / clamped_ratio

        temp_processed = Path(output_wav).with_suffix(".rb_temp.wav")

        # Rubberband command with time and pitch adjustments
        cmd = [
            "rubberband",
            "-q",
            "-t", f"{time_factor:.4f}",
            "-p", f"{pitch_semitones:.2f}",
            input_wav,
            str(temp_processed)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        src = temp_processed if res.returncode == 0 and temp_processed.exists() else Path(input_wav)

        data, sr = sf.read(str(src))
        target_samples = int(target_duration * sr)

        # Micro-fades (5ms in, 15ms out) for natural vocal release
        fade_in = int(0.005 * sr)
        fade_out = int(0.015 * sr)
        if len(data) > fade_in:
            data[:fade_in] *= np.linspace(0.0, 1.0, fade_in)

        if len(data) < target_samples:
            pad = np.zeros(target_samples - len(data), dtype=data.dtype)
            data = np.concatenate([data, pad])
        elif len(data) > target_samples:
            data = data[:target_samples]
            if len(data) > fade_out:
                data[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

        sf.write(output_wav, data, sr)
        if temp_processed.exists():
            temp_processed.unlink()

        return len(data) / sr

    def mix_scene(
        self,
        video_in: str,
        original_audio_wav: str,
        dubbed_dialogues: List[Dict[str, Any]],
        total_duration: float,
        output_mp4: str,
        original_duck_level: float = 0.18
    ):
        """
        Mixes dubbed dialogues with the original audio stem ducked
        so original room tone, foley, and score are preserved naturally.
        """
        sr = self.sample_rate
        master_len = int(total_duration * sr)
        master_dub = np.zeros(master_len, dtype=np.float32)

        # Load original audio for ducking bed
        orig_data, orig_sr = sf.read(original_audio_wav)
        if len(orig_data.shape) > 1:
            orig_data = orig_data.mean(axis=1) # to mono

        # Pad or trim original to master length
        if len(orig_data) < master_len:
            orig_bed = np.pad(orig_data, (0, master_len - len(orig_data)))
        else:
            orig_bed = orig_data[:master_len]

        # Dialogue ducking mask (1.0 where quiet, duck_level where talking)
        duck_mask = np.ones(master_len, dtype=np.float32)

        for d in dubbed_dialogues:
            start_sec = d["start"]
            wav_file = d["wav_path"]
            if not Path(wav_file).exists():
                continue
            data, d_sr = sf.read(wav_file)
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            start_sample = int(start_sec * sr)
            end_sample = min(master_len, start_sample + len(data))
            chunk = data[: end_sample - start_sample]

            # Place dialogue
            master_dub[start_sample:end_sample] += chunk

            # Duck original audio in speech regions (with 50ms ramp)
            ramp = int(0.05 * sr)
            s_ramp = max(0, start_sample - ramp)
            e_ramp = min(master_len, end_sample + ramp)
            duck_mask[s_ramp:e_ramp] = original_duck_level

        # Blend ducked ambient audio bed + crisp dubbed dialogue
        mixed_audio = (orig_bed * duck_mask) + (master_dub * 1.1)

        # Master limiter / normalizer
        peak = np.max(np.abs(mixed_audio))
        if peak > 0.95:
            mixed_audio = (mixed_audio / peak) * 0.95

        temp_dub_wav = Path(output_mp4).with_suffix(".final_mix.wav")
        sf.write(str(temp_dub_wav), mixed_audio, sr)

        # Remux video
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", str(temp_dub_wav),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if temp_dub_wav.exists():
            temp_dub_wav.unlink()
