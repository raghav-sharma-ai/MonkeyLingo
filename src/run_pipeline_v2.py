import os
import sys
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.ingest import AudioIngest
from src.gnani_client import GnaniClient
from src.gemini_dubber import GeminiDubber
from src.fit_and_mix import AudioFitter

def run_performance_dub():
    print("=" * 65)
    print("🎭 MonkeyLingo v2: Performance & Character Voice Dubbing")
    print("=" * 65)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    video_path = config["scene"]["video_path"]
    target_lang = config["project"]["target_language"]
    target_lang_name = config["project"]["target_language_name"]
    
    male_voice = config["characters"]["male_lead"]["voice"]     # Deepak
    female_voice = config["characters"]["female_lead"]["voice"] # Nalini
    
    print(f"[*] Input Video: {video_path}")
    print(f"[*] Robert (Male Lead) Voice: {male_voice}")
    print(f"[*] Mother (Female Lead) Voice: {female_voice}")
    print(f"[*] Target Language: {target_lang_name} ({target_lang})")

    # Step 1: Ingestion
    ingest = AudioIngest(video_path)
    master_wav = ingest.extract_audio()
    segments_file = Path("data/scene/segments.json")
    with open(segments_file, "r") as f:
        segments = json.load(f)

    # Step 2: Gemini Dubbing Director (Scripting with prosody, breath, pauses & pitch)
    gemini = GeminiDubber()
    # Remove older cached script if exists to re-run fresh director pass
    dramatic_cache = Path("cache") / f"dramatic_script_{target_lang_name}.json"
    if dramatic_cache.exists():
        dramatic_cache.unlink()

    script_items = gemini.analyze_and_script_scene(segments, target_lang_name)
    script_map = {item["id"]: item for item in script_items}

    # Step 3: Dual-Character Synthesis with Pitch & Speed Modulation
    gnani = GnaniClient()
    fitter = AudioFitter()

    out_takes_dir = Path("out/takes_v2")
    out_takes_dir.mkdir(parents=True, exist_ok=True)
    dubbed_items = []

    for seg in segments:
        line_id = seg["id"]
        orig_text = seg["text"]
        dur = seg["duration"]
        item = script_map.get(line_id, {})

        speaker = item.get("speaker", "Robert")
        voice_to_use = female_voice if speaker == "Mother" else male_voice
        hindi_script = item.get("dramatic_hindi_script", orig_text)
        emotion = item.get("emotion", "conversational")
        speed = float(item.get("suggested_speed", 1.0))
        pitch_st = float(item.get("suggested_pitch_semitones", 0.0))

        print(f"\n--- {line_id} ({dur:.2f}s) | Character: {speaker.upper()} ({voice_to_use}) ---")
        print(f"Original: \"{orig_text}\"")
        print(f"Emotion:  {emotion} | Speed: {speed:.2f}x | Pitch: {pitch_st:+.1f} st")
        print(f"Dialogue: \"{hindi_script}\"")

        raw_take = out_takes_dir / f"{line_id}_{speaker}_raw.wav"
        fitted_take = out_takes_dir / f"{line_id}_{speaker}_fitted.wav"

        try:
            gnani.tts_synthesize(
                text=hindi_script,
                voice=voice_to_use,
                language=target_lang,
                speed=speed,
                output_path=str(raw_take)
            )

            # Apply pitch modulation + timing fitting
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
            print(f"[✓] Take generated, pitch-modulated and fitted to {dur:.2f}s.")
        except Exception as e:
            print(f"[!] Error on {line_id}: {e}")

    # Step 4: Final Scene Mixdown (preserving room tone & ambient foley)
    print("\n[Step 4/4] Mixing final performance dub with background ambience ducking...")
    out_video = "out/final_dubbed_scene_v2.mp4"
    total_dur = fitter.get_duration(str(master_wav))
    fitter.mix_scene(
        video_in=video_path,
        original_audio_wav=str(master_wav),
        dubbed_dialogues=dubbed_items,
        total_duration=total_dur,
        output_mp4=out_video,
        original_duck_level=0.15
    )

    print("\n" + "=" * 65)
    print(f"🎉 Production-Grade Dub Exported to: {out_video}")
    print("=" * 65)

if __name__ == "__main__":
    run_performance_dub()
