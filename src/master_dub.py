import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
import soundfile as sf
import numpy as np

load_dotenv(".env")

from src.ingest import AudioIngest
from src.gnani_client import GnaniClient
from src.fit_and_mix import AudioFitter

def produce_master_dub():
    print("=" * 65)
    print("🎬 MonkeyLingo: Zero-Bleed Cinema Master Dub")
    print("=" * 65)

    video_path = "data/scene/scene.mp4"
    target_lang = "hi-IN"
    male_voice = "Deepak"   # STRICTLY 1 male voice for Robert
    female_voice = "Nalini" # STRICTLY 1 female voice for Mother

    print(f"[*] Robert (Male Lead): ONLY {male_voice} across all 11 lines")
    print(f"[*] Mother (Female Lead): ONLY {female_voice} across lines 001 & 002")

    # Load segments
    with open("data/scene/segments.json") as f:
        segments = json.load(f)

    # Load perfected script
    with open("cache/perfect_script.json") as f:
        script_items = json.load(f)

    gnani = GnaniClient()
    fitter = AudioFitter()

    out_takes_dir = Path("out/takes_v3")
    out_takes_dir.mkdir(parents=True, exist_ok=True)
    dubbed_items = []

    for item, seg in zip(script_items, segments):
        line_id = item["id"]
        speaker = item["speaker"]
        dur = item["duration"]
        voice = male_voice if speaker == "Robert" else female_voice
        hindi_text = item["hindi"]
        speed = item["speed"]
        pitch_st = item["pitch_semitones"]
        emotion = item["emotion"]

        print(f"\n--- {line_id} ({dur:.2f}s) | {speaker.upper()} -> Voice: {voice} ---")
        print(f"Emotion: {emotion} | Pitch: {pitch_st:+.1f} st | Speed: {speed:.2f}x")
        print(f"Hindi:   \"{hindi_text}\"")

        raw_take = out_takes_dir / f"{line_id}_{speaker}_raw.wav"
        fitted_take = out_takes_dir / f"{line_id}_{speaker}_fitted.wav"

        # Synthesize via Gnani Timbre
        gnani.tts_synthesize(
            text=hindi_text,
            voice=voice,
            language=target_lang,
            speed=speed,
            output_path=str(raw_take)
        )

        # Process take with rubberband pitch modulation and time-fitting
        fitter.process_take(
            input_wav=str(raw_take),
            target_duration=dur,
            pitch_semitones=pitch_st,
            output_wav=str(fitted_take)
        )

        dubbed_items.append({
            "id": line_id,
            "start": seg["start"],
            "wav_path": str(fitted_take)
        })

    # MIXING: ZERO BLEED FROM ENGLISH
    print("\n[Mixdown] Assembling clean dialogue track with zero English dialogue bleed...")
    sr = 48000
    orig_wav = "data/scene/master_audio_48k.wav"
    orig_data, orig_sr = sf.read(orig_wav)
    total_len = len(orig_data)

    # Create muted background: MUTE original audio during all dialogue periods
    ambient_bed = orig_data.copy()
    fade_len = int(0.03 * sr) # 30ms fade

    for s in segments:
        s_idx = max(0, int((s["start"] - 0.04) * sr))
        e_idx = min(total_len, int((s["end"] + 0.04) * sr))
        if s_idx + fade_len < e_idx:
            ambient_bed[s_idx:s_idx + fade_len] *= np.linspace(1.0, 0.0, fade_len)
            ambient_bed[s_idx + fade_len:e_idx - fade_len] = 0.0
            ambient_bed[e_idx - fade_len:e_idx] *= np.linspace(0.0, 1.0, fade_len)
        else:
            ambient_bed[s_idx:e_idx] = 0.0

    # Soften ambient bed level so dialogue is crystal clear
    ambient_bed *= 0.65

    # Assemble Hindi dialogue track
    dubbed_master = np.zeros(total_len, dtype=np.float32)
    for d in dubbed_items:
        data, _ = sf.read(d["wav_path"])
        s_idx = int(d["start"] * sr)
        e_idx = min(total_len, s_idx + len(data))
        dubbed_master[s_idx:e_idx] += data[:e_idx - s_idx]

    # Combine: Clean Hindi Dialogue + Ambience (English speech completely removed)
    final_master = ambient_bed + (dubbed_master * 1.05)

    # Limiter
    peak = np.max(np.abs(final_master))
    if peak > 0.95:
        final_master = (final_master / peak) * 0.95

    final_audio_wav = "out/final_clean_audio.wav"
    sf.write(final_audio_wav, final_master, sr)

    # Multiplex with video using ffmpeg
    final_output_mp4 = "out/MonkeyLingo_Master.mp4"
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", final_audio_wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        final_output_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("\n" + "=" * 65)
    print(f"🎉 MASTER VIDEO CREATED (ZERO BLEED): {final_output_mp4}")
    print("=" * 65)

if __name__ == "__main__":
    produce_master_dub()
